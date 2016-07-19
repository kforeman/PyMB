import numpy as np
import time
import warnings
import readline
from scipy.sparse import csc_matrix

def get_R_attr(obj, attr):
    '''
    Convenience function to return a named attribute from an R object (ListVector)
    e.g. get_R_attr(myModel.TMB.model, 'hessian') would return the equivalent of model$hessian
    '''
    return obj[obj.names.index(attr)]

class model:
    def __init__(self, name=None, filepath=None, codestr=None, **kwargs):
        '''
        Create a new TMB model, which utilizes an embedded R instance
        Optionally compile and load model upon instantiation if passing in filepath or codestr
        Parameters
        ----------
        name : str, default "TMB_{random.randint(1e10,9e10)}"
            Used to create model objects in R
        filepath : str (optional)
            Given a path to an existing .cpp file, the model will go ahead and compile it and load into R
        codestr : str (optional)
            A string containing .cpp code to be saved, compiled, and loaded into R
        **kwargs : optional
            Additional arguments passed to TMB_Model.compile()
        '''

        # make sure no hyphens in the model name, as that'll cause errors later
        if name:
            if name.find('-') != -1:
                raise Exception('"name" cannot contain hyphens.')

        # set model name
        self.name = name if name else 'TMB_{}'.format(np.random.randint(1e10,9e10))

        # initiate R session
        from rpy2 import robjects as ro
        self.R = ro

        # turn on numpy to R conversion
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()

        # create TMB link
        from rpy2.robjects.packages import importr
        self.TMB = importr('TMB')
        importr('Matrix')

        # create lists of data and initial parameter values for this model
        self.data = {}
        self.init = {}

        # compile if code passed
        if filepath or codestr:
            self.compile(filepath=filepath, codestr=codestr, **kwargs)

    def compile(self, filepath=None, codestr=None, output_dir='tmb_tmp',
        cc='g++', R='/usr/share/R/include',
        TMB='/usr/local/lib/R/site-library/TMB/include', LR='/usr/lib/R/lib', verbose=False, load=True):
        '''
        Compile TMB C++ code and load into R
        Parameters
        ----------
        filepath : str
            C++ code to compile
        codestr : str
            C++ code to save to .cpp then compile
        output_dir : str, default 'tmb_tmp'
            output directory for .cpp and .o
        cc : str, default 'g++'
            C++ compiler to use
        R : str, default '/usr/share/R/include'
            location of R shared library
            Note: R must be built with shared libraries
                  See http://stackoverflow.com/a/13224980/1028347
        TMB : str, default '/usr/local/lib/R/site-library/TMB/include'
            location of TMB library
        verbose : boolean, default False
            print compiler warnings
        load : boolean, default True
            load the model into Python after compilation
        '''
        # time compilation
        start = time.time()

        # check arguments
        if not filepath and not codestr:
            raise Exception('No filepath or codestr found.')

        # make the output directory if it doesn't already exist
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # if given just a filepath, copy the code into the output directory
        if filepath:
            self.filepath = os.path.join(output_dir, '{name}.cpp'.format(name=self.name))
            import shutil
            shutil.copy(filepath, self.filepath)
            if codestr:
                warnings.warn('Both filepath and codestr specified. Ignoring codestr.')

        # otherwise write code to file
        elif codestr:
            self.filepath = '{output_dir}/{name}.cpp'.format(output_dir=output_dir, name=self.name)

            # only rewrite cpp if identical code found
            if os.path.isfile(self.filepath) == False or file(self.filepath, 'r').read() != codestr:
                print('Saving model to {}.'.format(self.filepath))
                with file(self.filepath, 'w') as f:
                    f.write(codestr)
            else:
                print('Using {}.'.format(self.filepath))

        # compile the model
        # NOTE: cannot just call TMB.compile unfortunately - something about shared libraries not being hooked up correctly inside of embedded R sessions
        # TODO: skip recompiling when model has not changed
        import subprocess
        from pprint import pprint
        # compile cpp
        comp = '{cc} {include} {options} {f} -o {o}'.format(
            cc=cc,
            include='-I{R} -I{TMB}'.format(R=R, TMB=TMB),
            options='-DNDEBUG -DTMB_SAFEBOUNDS -DLIB_UNLOAD=R_unload_{} -fpic -O3 -pipe -g -c'.format(self.name),
            f='{output_dir}/{name}.cpp'.format(output_dir=output_dir, name=self.name),
            o='{output_dir}/{name}.o'.format(output_dir=output_dir, name=self.name))
        try:
            cmnd_output = subprocess.check_output(comp, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as exc:
            print(comp)
            print(exc.output)
            raise Exception('Your TMB code could not compile. See error above.')
        if verbose:
            print(comp)
            print(cmnd_output)
        # create shared object
        link = '{cc} {options} -o {so} {o} {link}'.format(
            cc=cc,
            options='-shared',
            so='{output_dir}/{name}.so'.format(output_dir=output_dir, name=self.name),
            o='{output_dir}/{name}.o'.format(output_dir=output_dir, name=self.name),
            link='-L{LR} -lR'.format(LR=LR))
        try:
            cmnd_output = subprocess.check_output(link, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as exc:
            print(link)
            print(exc.output)
            raise Exception('Your TMB code could not be linked. See error above.')

        # if a module of the same name has already been loaded, must unload R entirely it seems
        """
        TODO: fix this so R doesn't have to be restarted, potentially losing things the user has already loaded into R
        judging by https://github.com/kaskr/adcomp/issues/27 this should work:
        self.R.r('try(dyn.unload("{output_dir}/{name}.so"), silent=TRUE)'.format(output_dir=output_dir, name=self.name))
        but it doesn't - gives odd vector errors when trying to optimize
        """
        if self.name in [str(get_R_attr(i, 'name')[0]) for i in self.R.r('getLoadedDLLs()')]:
            warnings.warn('A model has already been loaded into TMB. Restarting R and reloading model to prevent conflicts.')
            self.R.r('sink("/dev/null")')
            self.R.r('try(dyn.unload("{output_dir}/{name}.so"), silent=TRUE)'.format(output_dir=output_dir, name=self.name))
            self.R.r('sink()')
            del self.R
            from rpy2 import robjects as ro
            self.R = ro
            del self.TMB
            from rpy2.robjects.packages import importr
            self.TMB = importr('TMB')

        # load the model into R
        if load:
            self.load_model(so_file='{output_dir}/{name}.so'.format(output_dir=output_dir, name=self.name))

        # output time
        print('Compiled in {:.1f}s.\n'.format(time.time()-start))

    def load_model(self, so_file=''):
        if so_file == '':
            so_file = 'tmb_tmp/{name}.so'.format(name=self.name)
        if not hasattr(self, 'filepath'):
            # assume that the cpp file is in the same directory with the same name if it wasn't specified
            self.filepath = so_file.replace('.so','.cpp')
        self.R.r('sink("/dev/null")')
        self.R.r('dyn.load("{so_file}")'.format(so_file=so_file))
        self.R.r('sink()')
        self.model_loaded = True

    def check_inputs(self, thing):
        import re
        missing = []
        with file(self.filepath) as f:
            for l in f:
                if re.match('^PARAMETER' if thing == 'init' else '^{}'.format(thing.upper()), l.strip()):
                    i = re.search(r"\(([A-Za-z0-9_]+)\)", l.strip()).group(1)
                    if i not in getattr(self, thing).keys():
                        missing.append(i)
        if missing:
            missing.sort()
            raise Exception('''Missing the following {thing}: {missing}\n
                Assign via e.g. myModel.{thing}["a"] = np.array(1., 2., 3.)'''.format(thing=thing, missing=missing))

    def build_objective_function(self, random=[], hessian=True, **kwargs):
        '''
        Builds the model objective function
        Parameters
        ----------
        random : list, default []
            which parameters should be treated as random effects (and thus integrated out of the likelihood function)
            can also be added manually via e.g. myModel.random = ['a','b']
        hessian : boolean, default True
            whether to calculate Hessian at optimum
        **kwargs : additional arguments to be passed to MakeADFun
        '''
        # first check to make sure everything necessary has been loaded
        if not hasattr(self, 'model_loaded'):
            raise Exception('Model not yet compiled/loaded. See TMB_model.compile().')
        self.check_inputs('data')
        self.check_inputs('init')

        # reload the model if it's already been built
        if hasattr(self, 'obj_fun_built'):
            try:
                del self.TMB.model
                self.R.r('dyn.load("{filepath}")'.format(filepath=self.filepath.replace('.cpp','.so')))
            except:
                pass

        # save the names of random effects
        if random or not hasattr(self, 'random'):
            random.sort()
            self.random = random

        # convert random effects to the appropriate format
        if self.random:
            kwargs['random'] = self.R.StrVector(self.random)

        # store a list of fixed effects (any parameter that is not random)
        self.fixed = [v for v in self.init.keys() if v not in self.random]

        # build the objective function
        self.TMB.model = self.TMB.MakeADFun(data=self.R.ListVector(self.data),
            parameters=self.R.ListVector(self.init), hessian=hessian, **kwargs)

        # set obj_fun_built
        self.obj_fun_built = True


    def optimize(self, opt_fun='nlminb', method='L-BFGS-B', draws=100, verbose=False, random=None, quiet=False, params=[], noparams=False, constrain=False, warning=True, **kwargs):
        '''
        Optimize the model and store results in TMB_Model.TMB.fit
        Parameters
        ----------
        opt_fun : str, default 'nlminb'
            the R optimization function to use (e.g. 'nlminb' or 'optim')
        method : str, default 'L-BGFS-B'
            method to use for optimization
        draws : int or Boolean, default 100
            if Truthy, will automatically simulate draws from the posterior
        verbose : boolean, default False
            whether to print detailed optimization state
        random : list, default []
            passed to PyMB.build_objective_function
            which parameters should be treated as random effects (and thus integrated out of the likelihood function)
            can also be added manually via e.g. myModel.random = ['a','b']
        params : list of strings, default []
            which parameters to simulate, defaults to [] which means all parameters
            list parameters by name to extract their posteriors from the model
        noparams : boolean, default False
            if True, will skip finding the means of the parameters entirely
        constrain : float or boolean, default False
            if float, will constrain any draws of a parameter to be within that many
                standard deviations of the median
        warning : bool
            print warning when there is non convergence with a model
        **kwargs : additional arguments to be passed to the R optimization function
        '''
        # time function execution
        start = time.time()

        # rebuild optimization function if new random parameters are given
        rebuild = False
        if random is not None:
            if not hasattr(self, 'random') or random != self.random:
                self.random = random
                rebuild = True

        # check to make sure the optimization function has been built
        if not hasattr(self.TMB, 'model') or rebuild:
            self.build_objective_function(random=self.random)

        # turn off warnings if verbose is not on
        if not verbose:
            self.R.r('''
                function(model) {
                    model$env$silent <- TRUE
                    model$env$tracemgc <- FALSE
                    model$env$inner.control$trace <- FALSE
                }
            ''')(self.TMB.model)

        # fit the model
        if quiet:
            self.R.r('sink("/dev/null")')
        self.TMB.fit = self.R.r[opt_fun](start=get_R_attr(self.TMB.model, 'par'), objective=get_R_attr(self.TMB.model, 'fn'),
            gradient=get_R_attr(self.TMB.model, 'gr'), method=method, **kwargs)
        if quiet:
            self.R.r('sink()')
        else:
            print('\nModel optimization complete in {:.1f}s.\n'.format(time.time()-start))

        # check for convergence
        self.convergence = self.TMB.fit[self.TMB.fit.names.index('convergence')][0]
        if warning and self.convergence != 0:
            print '\nThe model did not successfully converge, exited with the following warning message:'
            print self.TMB.fit[self.TMB.fit.names.index('message')][0] + '\n'

        # simulate parameters
        if not quiet:
            print('\n{}\n'.format(''.join(['-' for i in range(80)])))
        if not noparams:
            self.simulate_parameters(draws=draws, quiet=quiet, params=params, constrain=constrain)

    def report(self, name):
        '''
        Retrieve a quantity that has been reported within the model using e.g. REPORT(Y_hat);
        Parameters
        ----------
        name : str
            the name of the reported parameter/quantity to return
        '''
        return np.array(get_R_attr(get_R_attr(self.TMB.model, 'report')(), name))

    def simulate_parameters(self, draws=100, params=[], quiet=False, constrain=False):
        '''
        Simulate draws from the posterior variance/covariance matrix of the fixed and random effects
        Stores draws in Model.parameters dictionary
        Parameters
        ----------
        draws : int or boolean, default 1000
            if Truthy, number of correlated draws to simulate from the posterior
        params : list of strings, default []
            which parameters to simulate, defaults to [] which means all parameters
            list parameters by name to extract their posteriors from the model
        quiet : boolean, default False
            set to True in order to suppress outputting parameter summaries
        constrain : float or boolean, default False
            if float, will constrain any draws of a parameter to be within that many
                standard deviations of the median
        '''
        # time function
        start = time.time()

        # use all parameters if none specified
        if len(params) == 0:
            params = self.fixed + self.random

        # start storing parameters
        self.parameters = {}

        # run sdreport to get everything in the right format
        if not self.random:
            self.sdreport = self.TMB.sdreport(self.TMB.model, getJointPrecision=True,
                                              hessian_fixed=get_R_attr(self.TMB.model, 'he')())
        else:
            self.sdreport = self.TMB.sdreport(self.TMB.model, getJointPrecision=True)

        # extraction convenience function
        # filters the object down to just a list of desired parameters
        extract_params = self.R.r('''function(obj, nm, keepers) {
            ii <- nm %in% keepers;
            if (is.vector(obj)) {
                return(obj[ii]);
            } else {
                return(obj[ii,ii]);
            };
        }''')

        # extract the joint precision matrix
        joint_prec_full = get_R_attr(self.sdreport, 'jointPrecision')
        joint_prec_names = self.R.r('row.names')(joint_prec_full)
        joint_prec_R = extract_params(joint_prec_full, joint_prec_names, params)

        # convert joint precision matrix to scipy.sparse.csc_matrix
        # (bypassing dense conversion step that happens if you use rpy2)
        def get_R_slot(obj, slot):
            return np.array(self.R.r('function(obj) {{ return(obj@{}) }}'.format(slot))(obj))
        joint_prec = csc_matrix((get_R_slot(joint_prec_R, 'x'),
                                 get_R_slot(joint_prec_R, 'i'),
                                 get_R_slot(joint_prec_R, 'p')))

        # make a list of all the parameters, ordered by position in the joint precision matrix
        ordered_params = np.array(self.R.r('row.names')(joint_prec_R))

        ### extract fixed effects
        # means
        fixed_mean_raw = get_R_attr(self.sdreport, 'par.fixed')
        # sds
        fixed_sd_raw = get_R_attr(self.sdreport, 'cov.fixed')
        # names
        fixed_names = fixed_mean_raw.names
        # keep just selected
        fixed_mean = extract_params(fixed_mean_raw, fixed_names, params)
        fixed_sd = extract_params(self.R.r('sqrt')(fixed_sd_raw), fixed_names, params)

        ### extract random effects
        # means
        ran_mean_raw = get_R_attr(self.sdreport, 'par.random')
        # sds
        ran_sd_raw = get_R_attr(self.sdreport, 'diag.cov.random')
        # names
        ran_names = ran_mean_raw.names
        # keep just selected
        ran_mean = extract_params(ran_mean_raw, ran_names,  params)
        ran_sd = extract_params(self.R.r('sqrt')(ran_sd_raw), ran_names, params)

        ### put the means/sds in the right order
        ## the reason being that the joint precision matrix is based on order of model specification, ignoring random vs fixed
        # initialize R vectors
        means = self.R.FloatVector([self.R.NA_Real for i in xrange(len(ordered_params))])
        sds = self.R.FloatVector([self.R.NA_Real for i in xrange(len(ordered_params))])
        # loop through and add the parameters to the means/sds in the correct order
        for p in set(ordered_params):
            # index in joint precision matrix
            i_joint = [ii for ii,pp in enumerate(ordered_params) if pp == p]
            # index in random effects
            i_ran = [ii for ii,pp in enumerate(ran_names) if pp == p]
            # index in fixed effects
            i_fixed = [ii for ii,pp in enumerate(fixed_names) if pp == p]
            # copy into the appropriate position
            if i_ran:
                means[i_joint[0]:i_joint[-1]+1] = ran_mean[i_ran[0]:i_ran[-1]+1]
                sds[i_joint[0]:i_joint[-1]+1] = ran_sd[i_ran[0]:i_ran[-1]+1]
            elif i_fixed:
                means[i_joint[0]:i_joint[-1]+1] = fixed_mean[i_fixed[0]:i_fixed[-1]+1]
                sds[i_joint[0]:i_joint[-1]+1] = fixed_sd[i_fixed[0]:i_fixed[-1]+1]

        ### generate draws
        # convert mean/sd to python
        means = np.array(means)
        sds = np.array(sds)
        ## simulation function
        # see http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        def gen_draws(mu, prec, n):
            from scikits.sparse.cholmod import cholesky
            from scipy.sparse.linalg import spsolve
            z = np.random.normal(size=(mu.shape[0],n))
            chol_jp = cholesky(prec)
            ### note: would typically use scikits.sparse.cholmod.cholesky.solve_Lt,
            ### but there seems to be a bug there: https://github.com/njsmith/scikits-sparse/issues/9#issuecomment-76862652
            return mu[:,np.newaxis] + chol_jp.apply_Pt(spsolve(chol_jp.L().T, z))
        #### make draws
        if draws:
            param_draws = gen_draws(means, joint_prec, draws)

        ### store results
        # constrain draws if requested
        if constrain and draws:
            # find which draws are more than {constraint} standard deviations from the mean
            wacky_draws = np.where(np.any([ \
                          np.greater(param_draws.T, means + (constrain * sds)), \
                          np.less(param_draws.T, means - (constrain * sds))], axis=0).T)
            # replace those draws with the mean
            param_draws[wacky_draws] = means[wacky_draws[0]]
        # add parameters' mean, sd, and optionally draws to the parameters dictionary
        for p in set(ordered_params):
            i = [ii for ii,pp in enumerate(ordered_params) if pp == p] # names will be the same for every item in a vector/matrix, so find all corresponding indices
            if draws:
                if type(self.init[p]) == np.ndarray:
                    these_draws = param_draws[i,].reshape(list(self.init[p].shape) + [draws], order='F')
                else:
                    these_draws = param_draws[i,]
                self.parameters[p] = {
                    'mean': means[i],
                    'sd': sds[i],
                    'draws': these_draws
                }
            else:
                self.parameters[p] = {
                    'mean': means[i],
                    'sd': sds[i]
                }
        # store "raw" results
        self._params = {
            'sd': sds,
            'mean': means,
            'joint_prec': joint_prec,
            'name': ordered_params
        }

        ### print results
        if not quiet:
            print('\nSimulated {n} draws in {t:.1f}s.\n'.format(n=draws, t=time.time()-start))
            self.print_parameters()

    def draws(self, parameter):
        '''
        Convenience function to return the draws for a specific parameter as a numpy.ndarray
        '''
        return self.parameters[parameter]['draws']

    def print_parameters(self):
        '''
        Print summary statistics of the model parameter fits
        '''
        np.set_printoptions(threshold=5, edgeitems=2)
        for p,v in self.parameters.iteritems():
            if 'draws' in v:
                d = v['draws']
                if d.shape[0] == 1:
                    print('{p}:\n\tmean\t{m}\n\tsd\t{s}\n\tdraws\t{d}\n\tshape\t{z}'.format(p=p, m=v['mean'], s=v['sd'], d=d, z=d.shape))
                elif len(d.shape) == 2:
                    print('{p}:\n\tmean\t{m}\n\tsd\t{s}\n\tdraws\t{d}\n\tshape\t{z}'.format(p=p, m=v['mean'], s=v['sd'], d='[{0},\n\t\t ...,\n\t\t {1}]'.format(d[0,:], d[d.shape[0]-1,:]), z=d.shape))
                else:
                    print('{p}:\n\tmean\t{m}\n\tsd\t{s}\n\tdraws\t{d}\n\tshape\t{z}'.format(p=p, m=v['mean'], s=v['sd'], d='[[{0},\n\t\t ...,\n\t\t {1}]]'.format(d[0,0,:], d[d.shape[0]-1,d.shape[1]-1,:]), z=d.shape))
            else:
                print('{p}:\n\tmean\t{m}\n\tsd\t{s}\n\tdraws\tNone'.format(p=p, m=v['mean'], s=v['sd']))
        np.set_printoptions(threshold=1000, edgeitems=3)

