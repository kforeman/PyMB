import numpy as np
import time
import warnings

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

        # create lists of data and initial parameter values for this model
        self.data = {}
        self.init = {}

        # compile if code passed
        if filepath or codestr:
            self.compile(filepath=filepath, codestr=codestr, **kwargs)

    def compile(self, filepath=None, codestr=None, output_dir='tmb_tmp',
        cc='g++', R='/usr/share/R/include',
        TMB='/usr/local/lib/R/site-library/TMB/include'):
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
        '''
        # time compilation
        start = time.time()

        # check arguments
        if not filepath and not codestr:
            raise Exception('No filepath or codestr found.')

        # if given just a filepath, simply store the path
        if filepath:
            self.filepath = filepath
            if codestr:
                warnings.warn('Both filepath and codestr specified. Ignoring codestr.')

        # otherwise write code to file
        elif codestr:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
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
        # create shared object
        link = '{cc} {options} -o {so} {o} {link}'.format(
            cc=cc,
            options='-shared',
            so='{output_dir}/{name}.so'.format(output_dir=output_dir, name=self.name),
            o='{output_dir}/{name}.o'.format(output_dir=output_dir, name=self.name),
            link='-L/usr/lib/R/lib -lR')
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
            del self.R
            del ro
            from rpy2 import robjects as ro
            self.R = ro
            del self.TMB
            from rpy2.robjects.packages import importr
            self.TMB = importr('TMB')


        # load the model into R
        self.R.r('dyn.load("{output_dir}/{name}.so")'.format(output_dir=output_dir, name=self.name))
        self.model_loaded = True

        # output time
        print('Compiled in {:.1f}s.\n'.format(time.time()-start))

    def check_inputs(self, thing):
        import re
        missing = []
        with file(self.filepath) as f:
            for l in f:
                if re.match('^PARAMETER' if thing == 'init' else '^{}'.format(thing.upper()), l.strip()):
                    i = re.search(r"\(([A-Za-z]+)\)", l.strip()).group(1)
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


    def optimize(self, opt_fun='nlminb', method='L-BFGS-B', draw=100, **kwargs):
        '''
        Optimize the model and store results in TMB_Model.TMB.fit

        Parameters
        ----------
        opt_fun : str, default 'nlminb'
            the R optimization function to use (e.g. 'nlminb' or 'optim')
        method : str, default 'L-BGFS-B'
            method to use for optimization
        draw : int or Boolean, default 100
            if Truthy, will automatically simulate draws from the posterior
        **kwargs: additional arguments to be passed to the R optimization function
        '''
        # time function execution
        start = time.time()

        # check to make sure the optimization function has been built
        if not hasattr(self.TMB, 'model'):
            self.build_objective_function()

        # fit the model
        self.TMB.fit = self.R.r[opt_fun](start=get_R_attr(self.TMB.model, 'par'), objective=get_R_attr(self.TMB.model, 'fn'),
            gradient=get_R_attr(self.TMB.model, 'gr'), method=method, **kwargs)
        print('\nModel optimization complete in {:.1f}s.\n'.format(time.time()-start))

        # simulate parameters
        if draw:
            print('\n{}\n'.format(''.join(['-' for i in range(80)])))
            if type(draw) == int:
                self.simulate_parameters(n = draw)
            else:
                self.simulate_parameters()

    def report(self, name):
        '''
        Retrieve a quantity that has been reported within the model using e.g. REPORT(Y_hat);

        Parameters
        ----------
        name : str
            the name of the reported parameter/quantity to return
        '''
        return np.array(get_R_attr(get_R_attr(self.TMB.model, 'report')(), name))

    def simulate_parameters(self, n=100):
        '''
        Simulate draws from the posterior variance/covariance matrix of the fixed and random effects

        Stores draws in TMB_Model.parameters dictionary

        Parameters
        ----------
        n : int, default 1000
            number of draws to make
        '''
        # time function
        start = time.time()

        # start storing parameters
        self.parameters = {}

        # run sdreport to get everything in the right format
        if not self.random:
            self.sdreport = self.TMB.sdreport(self.TMB.model, getJointPrecision=True, hessian_fixed=get_R_attr(self.TMB.model, 'he')())
        else:
            self.sdreport = self.TMB.sdreport(self.TMB.model, getJointPrecision=True)

        # fixed effects only models
        if self.fixed and not self.random:
            # means
            fixed_mean = np.array(get_R_attr(self.sdreport, 'par.fixed'))
            # precision matrix (note: for fixed effects, the var/cov matrix is returned so must invert it)
            fixed_vcov = np.array(get_R_attr(self.sdreport, 'cov.fixed'))
            fixed_prec = np.linalg.inv(fixed_vcov)
            # sd
            fixed_sd = np.sqrt(np.diag(fixed_vcov))
            # draws (http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution)
            z = np.random.normal(size=(fixed_mean.shape[0],n))
            L_inv = np.linalg.cholesky(fixed_prec)
            fixed_draws = fixed_mean.reshape((fixed_mean.shape[0], 1)) + np.linalg.solve(L_inv, np.linalg.solve(L_inv.T, z))
            # save results
            names = get_R_attr(self.sdreport, 'par.fixed').names
            for m in set(names):
                i = [ii for ii,mm in enumerate(names) if mm == m] # names will be the same for every item in a vector/matrix, so find all corresponding indices
                self.parameters[m] = {
                    'mean': fixed_mean[i],
                    'sd': fixed_sd[i],
                    'draws': fixed_draws[i,]
                }

        # random effects models
        else:
            from scikits.sparse.cholmod import cholesky
            from scipy.sparse import csc_matrix
            # means
            ran_mean = np.array(get_R_attr(self.sdreport, 'par.random'))
            fixed_mean = np.array(get_R_attr(self.sdreport, 'par.fixed'))
            # joint precision matrix
            joint_prec = csc_matrix(self.R.r['as.matrix'](get_R_attr(self.sdreport, 'jointPrecision')))
            # sd
            fixed_sd = np.sqrt(np.diag(np.array(get_R_attr(self.sdreport, 'cov.fixed'))))
            ran_sd = np.sqrt(np.array(get_R_attr(self.sdreport, 'diag.cov.random')))
            # find names of parameters on joint precision matrix
            joint_names = self.R.r['row.names'](get_R_attr(self.sdreport, 'jointPrecision'))
            ran_names = get_R_attr(self.sdreport, 'par.random').names
            fixed_names = get_R_attr(self.sdreport, 'par.fixed').names
            # sort means appropriately
            means = np.empty(shape=(len(joint_names),1))
            sds = np.empty(shape=(len(joint_names),1))
            for m in set(joint_names):
                # index in joint
                i_joint = [ii for ii,mm in enumerate(joint_names) if mm == m]
                if m in ran_names:
                    # index in random effects vectors
                    i_ran = [ii for ii,mm in enumerate(ran_names) if mm == m]
                    means[i_joint] = ran_mean[i_ran]
                    sds[i_joint] = ran_sd[i_ran]
                else:
                    # index in fixed effects vectors
                    i_fixed = [ii for ii,mm in enumerate(fixed_names) if mm == m]
                    means[i_joint] = fixed_mean[i_fixed]
                    sds[i_joint] = fixed_sd[i_fixed]
            # draws (http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution)
            z = np.random.normal(size=(means.shape[0],n))
            L_inv = cholesky(joint_prec)
            draws = means + L_inv.solve_LDLt(z)
            # save results
            for m in set(joint_names):
                i = [ii for ii,mm in enumerate(joint_names) if mm == m] # names will be the same for every item in a vector/matrix, so find all corresponding indices
                self.parameters[m] = {
                    'mean': means[i],
                    'sd': sds[i],
                    'draws': draws[i,]
                }
        print('\nSimulated {n} draws in {t:.1f}s.\n'.format(n=n, t=time.time()-start))
        self.print_parameters()

    def print_parameters(self):
        '''
        Print summary statistics of the model parameter fits
        '''
        np.set_printoptions(threshold=5, edgeitems=2)
        for p,v in self.parameters.iteritems():
            print('{p}:\n\tmean\t{m}\n\tsd\t{s}\n\tdraws\t{d}\n\tshape\t{z}'.format(p=p, m=v['mean'], s=v['sd'], d=v['draws'], z=v['draws'].shape))
        np.set_printoptions(threshold=1000, edgeitems=3)
