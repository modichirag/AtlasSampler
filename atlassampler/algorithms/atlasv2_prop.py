import sys
import numpy as np
from scipy.stats import multivariate_normal

from ..util import Sampler, PrintException
from .stepadapt_samplers import DRHMC_AdaptiveStepsize

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()

__all__ = ["Atlasv2_Prop"]

class Atlasv2_Prop(DRHMC_AdaptiveStepsize):
    """Atlas is "Adaptive Trajectory Length And Stepsize" sampler. 
    
    Warmup: 
    In the warmup phase, Atlas first adapts the stepsize and the trajectory lenght as follows:
        1. Baseline stepsize is adapted with dual averaging to target an acceptance rate of ``target_accept``.
        2. Next, Atlas constructs an empirical distribution of trajectory lengths (edist_traj).
        It run ``n_leapfrog_adapt`` iterations upto U-turn in every chain and stores the trajectory length.
        Then it combines this information from all chains, removes the lenghts that are too short or too long
        (outside ``low_nleap_percentile`` and ``high_nleap_percentile``) and saves the remaining lengths
        to construct edist_traj.

    Sampling:
    Every iteration of Atlas consists of 2 stages.

        1. Preliminary stage
        Atlas first runs a preliminary trajectory with baseline step_size uptil U-turn.
        If the number_of_steps_to_u-turn > ``min_nleapfrog`` (default = 3), it declares a failure and skips to
        `delayed_step_upon_failure`. Otherwise, it follows GIST algorithm to make a proposal in the
        [``offset``, 1) fraction of the trajectory. If this is accepted, algorithm moves to next iteration. 
        If not, then it moves to delayed stage.

        2. Delayed stage
        There are two types of delayed stages depending on the preliminary stage outcome:
        a. Delayed step upon failure:
            This is executed when number_of_steps_to_u-turn > min_nleapfrog, indicating bad stepsize.
            This step locally adapts stepsize by constructing an approximate local Hessian.
            The trajectory length is sampled from the empirical distribution (edist_traj) constructed in warmup.
            If ``constant_trajectory`` != 0, the number of leapfrog steps is scaled by the ratio of stepsize
            to baseline stepsize.
        b. Delayed step:
            This is executed when the preliminary step is rejected for the GIST proposal.
            If ``probabilistic`` == 0, then the algorithm always makes this delayed proposal.
            If ``probabilistic`` == 1, then the algorithm makes this delayed proposal only if the preliminary stage is rejected
            *not due to a sub u-turn.*
            In this stage, the stepsize is locally adapted as the previous case. However the number of leaprog steps
            is determined by the number of steps to the proposal made in the preliminary stage. If ``constant_trajectory`` != 2,
            the number of leapfrog steps is kept the same. Otherwise it is scaled by the ratio of stepsize to baseline stepsize.
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, 
                 offset=None, 
                 constant_trajectory=0,
                 probabilistic=0,
                 delayed_proposals=1,
                 low_nleap_percentile=10, 
                 high_nleap_percentile=90, 
                 **kwargs):
        """Initialize an instance of Atlas.
        
        Args:
            D (int): dimensionality of the parameter space
            log_prob (function): function that takes in parameter value and returns log_probability
            grad_log_prob (function):  function that takes in parameter value 
                and returns the gradient of the log_probability
            offset (float in range 0-1, default=None): sample the proposal from [offset, 1) fraction
                of the no-U turn trajectory. If None, offset is shuffled and randomly picked
                between [0.33, 0.66] for every iteration.      
            constant_trajectory (0, 1, or 2, default=1): determine if number of leapfrog steps for
                delayed proposals are scaled by stepsize ratio. 0: no scaling for any delayed proposal. 
                1: only delayed proposal on failure are scaled. 2: both delayed proposals are scaled 
            probabilistic (0 or 1, default=0): wether to make delayed proposal without failure
                0: always make second delayed proposal. 1: make delayed proposal only the first proposal
                is rejected not due to a sub u-turn  
        
        Keyword args:
            min_nleapfrog (int, default=3): minimum number of leapfrog steps.
                use delayed_rejection_with_failure if number of steps in first proposal is less that it.
            max_nleapfrog (int, default=1024): maximum number of leapfrog steps in any iteration.
            n_hessian_samples (int, default=10): minimum number of points needed to evaluate Hessian
            n_hessian_attempts (int, default=10): maximum number of attempts to evaluate Hessian
                by reducing step-size with a factor of 2 on each attempt. 
            low_nleap_percentile (int, default=10): percentile below which u-turn trajectories
                in the empirical trajectory length distribution of warmup are discarded.
            high_nleap_percentile (int, default=50): percentile above which u-turn trajectories
                in the empirical trajectory length distribution of warmup are discarded.
            nleap_factor (float, default=1.): scale the empirical distribution of trajectory lengths.
            max_stepsize_reduction (float, default=500): minimum possible step-size is set to be the
                baseline stepsize/max_stepsize_reduction
        """

        super(Atlasv2_Prop, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, 
                                    mass_matrix=mass_matrix, offset=offset,
                                    low_nleap_percentile=low_nleap_percentile,
                                    high_nleap_percentile=high_nleap_percentile, 
                                    **kwargs)
        if self.min_nleapfrog <= 2:
            print("min nleapfrog should at least be 2 for delayed proposals")
            raise
        self.probabilistic = probabilistic
        self.delayed_proposals = delayed_proposals
        self.constant_trajectory=constant_trajectory

        
    def preliminary_step(self, q, p, step_size, offset, n_leapfrog_input=None):
        """
        First step is making a no-uturn proposal.
        """
        try:
            # Go forward
            Nuturn, qlist, plist, glist, success = self.nuts_criterion(q, p, step_size)
            # if not success:
                # print(success, Nuturn, q[0])
                # if Nuturn >= self.min_nleapfrog :
                #     print(success, Nuturn, q[0])  

        except Exception as e:
            PrintException()
            log_prob_list = [-np.inf, -np.inf, -np.inf] # log prob, log prob H, log prob N
            qpg_list = [[], [], []]
            Hs, n_leapfrog = [0, 0], 0
            return [q, p], qpg_list, log_prob_list, Hs, n_leapfrog

        if Nuturn < self.min_nleapfrog :
            log_prob_list = [-np.inf, -np.inf, -np.inf]
            Hs, n_leapfrog = [0, 0], 0.
            return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog

        else:
            n_leapfrog, lp_N = self.nleapfrog_sample_and_lp(Nuturn,  offset, nleapfrog=n_leapfrog_input)
            if lp_N == -np.inf:
                log_prob_list, Hs = [0., 0., 0.], [0., 0.]
                n_leapfrog = 0
                return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog            
            
            else:
                if (n_leapfrog_input is not None) :
                    assert  n_leapfrog == n_leapfrog_input
                q1, p1 = qlist[n_leapfrog], plist[n_leapfrog]           

                # Go backward
                Nuturn_rev = self.nuts_criterion(q1, -p1, step_size)[0]
                n_leapfrog_rev, lp_N_rev = self.nleapfrog_sample_and_lp(Nuturn_rev, offset, nleapfrog=n_leapfrog)
                assert n_leapfrog_rev == n_leapfrog
                self.Vgcount -= min(Nuturn_rev, n_leapfrog) #adjust for number of common steps when going backward
                
                # Hastings
                log_prob_H, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
                Hs  = [H0, H1]
                log_prob_N = lp_N_rev - lp_N                 
                log_prob_accept = log_prob_H + log_prob_N
                if np.isnan(log_prob_accept):
                    log_prob_accept = -np.inf
                log_prob_accept = min(0, log_prob_accept) 
                log_prob_list = [log_prob_accept, log_prob_H, log_prob_N]
                        
                return [q1, p1], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog


        
    def delayed_step(self, q0, p0, qlist, glist, step_size, offset, n_leapfrog, log_prob_list_first): 
        ##
        
        log_prob_accept_first, log_prob_H_first, log_prob_N_first = log_prob_list_first
        
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            epsf1 = self.get_stepsize_distribution(q0, p0, qlist, glist, step_size)[1]
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            if self.constant_trajectory == 2:
                #n_leapfrog_new = int(min(self.max_nleapfrog, n_leapfrog*step_size / step_size_new))  ##THIS WAS CHANGED TO TEST rosenbrockhy3. How does it affect others?
                n_leapfrog_new = int(n_leapfrog*step_size / step_size_new)
            else:
                n_leapfrog_new = n_leapfrog                
            q1, p1, _, _ = self.leapfrog(q0, p0, n_leapfrog_new, step_size_new)
            log_prob_H, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            
            # Ghost trajectory for the second proposal
            qp_ghost, qpg_ghost_list, log_prob_list_ghost, Hs_ghost, n_leapfrog_ghost = \
                                        self.preliminary_step(q1, -p1, step_size, offset=offset, n_leapfrog_input=n_leapfrog)

            if n_leapfrog_ghost == 0: #in this case, ghost stage cannot lead to the current delayed propsal
                qf, pf, accepted = q0, p0, -14

            else:
                log_prob_accept_ghost, log_prob_H_ghost, log_prob_N_ghost = log_prob_list_ghost
                if (self.probabilistic >= 1) & (log_prob_N_ghost == -np.inf): 
                        qf, pf, accepted = q0, p0, -12
                else:
                    # Hastings
                    qlist_ghost, plist_ghost, glist_ghost = qpg_ghost_list
                    epsf2 = self.get_stepsize_distribution(q1, -p1, qlist_ghost, glist_ghost, step_size)[1]
                    log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
                    log_prob_delayed = np.log((1-np.exp(log_prob_accept_ghost))) - np.log((1- np.exp(log_prob_accept_first)))
                    log_prob_proposal_first = log_prob_N_first
                    log_prob_proposal_ghost = log_prob_N_ghost
                    log_prob_proposal = log_prob_proposal_ghost - log_prob_proposal_first # this is different from atlasv2
                    if self.probabilistic == 2:
                        log_prob_proposal_first = np.log(1 - np.exp(log_prob_H_first))
                        log_prob_proposal_ghost = np.log(1 - np.exp(log_prob_H_ghost))
                        log_prob_proposal += log_prob_proposal_ghost - log_prob_proposal_first

                    log_prob_accept = log_prob_H + log_prob_delayed + log_prob_eps + log_prob_proposal

                    u =  np.random.uniform(0., 1., size=1)
                    if np.isnan(log_prob_accept) or (q0-q1).sum()==0:
                        qf, pf, accepted = q0, p0, -198
                    elif  np.log(u) > min(0., log_prob_accept):
                        qf, pf = q0, p0
                        if log_prob_eps == -np.inf: accepted = -13
                        else: accepted = -11
                    else: 
                        qf, pf, accepted = q1, p1, 2

            return qf, pf, accepted, [H0, H1]
            
        except Exception as e:
            PrintException()
            return q0, p0, -199, [0, 0]
        

    def delayed_step_upon_failure(self, q0, p0, step_size): 
        """Delayed step upon failure is executed when Nuturn of first step < min_nleapfrog.
        """
        try:
            # Estimate the Hessian and the stepsize given the rejected trajectory
            epsf1 = self.get_stepsize_distribution(q0, p0, [], [], step_size)[1]
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            n_leapfrog_new =  self.nleapfrog_jitter_dist(step_size)
            if self.constant_trajectory != 0:
                n_leapfrog_new = n_leapfrog_new*step_size / step_size_new
            n_leapfrog_new = int(min(self.max_nleapfrog, max(self.min_nleapfrog, n_leapfrog_new)))
                    
            q1, p1, _, _ = self.leapfrog(q0, p0, n_leapfrog_new, step_size_new)
            log_prob_H, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            Hs = [H0, H1]
            
            # Ghost trajectory for the second proposal
            try:
                Nuturn_ghost = self.nuts_criterion(q1, -p1, step_size)[0]            
                if Nuturn_ghost >= self.min_nleapfrog :
                    qf, pf, accepted = q0, p0, -24
                    return qf, pf, accepted, Hs                
            except Exception as e:
                PrintException()                

            # Hastings
            epsf2 = self.get_stepsize_distribution(q1, -p1, [], [], step_size)[1]
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob_accept = log_prob_H + log_prob_eps

            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob_accept) or (q0-q1).sum()==0:
                qf, pf, accepted = q0, p0, -298
            elif  np.log(u) > min(0., log_prob_accept):
                qf, pf = q0, p0
                if log_prob_eps == -np.inf:
                    accepted = -23
                else:
                    accepted = -21
            else: 
                qf, pf, accepted = q1, p1, 3
                    
            return qf, pf, accepted, [H0, H1]
            
        except Exception as e:
            PrintException()
            return q0, p0, -299, [0, 0]
        

        
    def step(self, q, n_leapfrog=None, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        offset = self.offset_function()
        qp, qpg_list, log_prob_list, Hs, n_leapfrog = self.preliminary_step(q, p, step_size, offset=offset)
        log_prob_accept, log_prob_H, log_prob_N = log_prob_list
        q1, p1 = qp
        qlist, plist, glist = qpg_list
        
        # Hastings
        if n_leapfrog == 0 :
            return self.delayed_step_upon_failure(q, p, step_size=step_size)
        else:
            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob_accept): log_prob_accept = -np.inf
            if  np.log(u) > min(0., log_prob_accept): #reject
                if self.delayed_proposals:
                    if self.probabilistic == 1:
                        if log_prob_N == -np.inf:
                            return q, p, -1, Hs
                        else:
                            return self.delayed_step(q, p, qlist, glist,
                                                     step_size = step_size, offset = offset, n_leapfrog = n_leapfrog,
                                                     log_prob_list_first = log_prob_list)
                                                     # log_prob_accept_first=log_prob_accept, log_prob_H_first=log_prob_H, log_prob_N_first=log_prob_N)
                    elif self.probabilistic == 2:
                        if log_prob_N == -np.inf:
                            return q, p, -1, Hs
                        else:
                            v =  np.random.uniform(0., 1., size=1)
                            log_prob_proposal = np.log(1 - np.exp(log_prob_H))
                            if np.isnan(log_prob_proposal): log_prob_proposal == -np.inf
                            if np.log(v) > min(0., log_prob_proposal):
                                return q, p, -1, Hs                                
                            else:
                                return self.delayed_step(q, p, qlist, glist,
                                                         step_size = step_size, offset = offset, n_leapfrog = n_leapfrog,
                                                         log_prob_list_first = log_prob_list)

                    else:
                        return self.delayed_step(q, p, qlist, glist, 
                                                 step_size = step_size, offset = offset, n_leapfrog = n_leapfrog,
                                                 log_prob_list_first = log_prob_list)
                else:
                    return q, p, -1, Hs

            else: #accept
                qf, pf = q1, p1
                accepted = 1
                return qf, pf, accepted, Hs
        
            
    def sample(self, q, p=None,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, n_leapfrog_adapt=100,
               target_accept=0.65, 
               seed=99,
               comm=None,
               verbose=False):
    
        state = Sampler()
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose

        if n_stepsize_adapt:
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        if n_leapfrog_adapt:  # Construct a distribution of trajectory lengths
            q = self.adapt_trajectory_length(q, n_leapfrog_adapt)
            self.combine_trajectories_from_chains(comm)
            state.trajectories = self.traj_array
            print(f"Shape of trajectories : ", self.traj_array.shape)
            self.nleapfrog_jitter()
        else:
            self.nleapfrog_jitter_dist = lambda x:  self.n_leapfrog
            
        for i in range(n_burnin):  # Burnin
            q, p, accepted, Hs = self.step(q) 

        for i in range(n_samples):  # Sample
            state.i += 1
            q, p, accepted, Hs = self.step(q) 
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            if (i%(n_samples//10) == 0):
                print(f"Iteration {i} of {n_samples}")

        state.to_array()
        return state
