## Model imports

# Deterministic dynamics model
# Deterministic direct model

## QD imports

def random_archive_init_model(params, to_evaluate):
    for i in range(0, params['random_init_batch']):
        x = np.random.uniform(low=params['min'], high=params['max'],size=params['dim_x'])
        to_evaluate += [(x, self.f_model)]
        
    return to_evaluate

def diverse_archive_init(params):
    """
    Return an archive of diverse behaviors found on a model
    """
    
    ## Instantiate a model

    ## Initialize the weights of the model

    ## Create an archive filled with randomly parameterized individuals
    to_model_evaluate = []
    archive = random_archive_init_model(params)
    
    ## Perform the QD search on the model ## A revoir
    start = time.time()
    add_list_model_final = []
    all_model_eval = []
    gen = 0
    while len(add_list_model_final) < params['min_found_model']:
    #for i in range(5000): # 600 generations (500 gens = 100,000 evals)
        to_model_evaluate=[]

        if (len(self.model_archive) <= params['random_init']*self.n_niches) \
           and params['init_method'] != 'vanilla':
            to_model_evaluate = self.random_archive_init_model(to_model_evaluate)
            if len(self.model_archive) > 0:
                to_model_evaluate = random.sample(to_model_evaluate,
                                                  int(params['random_init']*self.n_niches -
                                                      len(self.model_archive)))


                ## Create fake archive with new species created to have an archive size of 100
                fake_archive = self.model_archive.copy()
                for n in range(int(params['random_init']*self.n_niches -
                                                      len(self.model_archive))):
                    s = cm.Species(to_model_evaluate[n][0], [], [])
                    fake_archive.append(s)

                to_model_evaluate = self.select_and_mutate(to_model_evaluate,
                                                           fake_archive,
                                                           self.f_model, params)
        else:
            to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive,
                                                       self.f_model, params)
        if params["model_variant"]=="dynamics" or params["perfect_model_on"]:
            #s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)
            print("Starting parallel evaluation of individuals")
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            print("Finished parallel evaluation of individuals")
        elif params["model_variant"]=="direct":
            s_list_model = self.serial_eval(evaluate_, to_model_evaluate, params)
        elif params["model_variant"]=="all_dynamics":
            s_list_model = model_evaluate_all_(to_model_evaluate)
        #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
        self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)

        add_list_model_final += add_list_model
        all_model_eval += to_model_evaluate # count all inds evaluated by model
        #print("to model eval length: ",len(to_model_evaluate)) 
        #print("s list length: ",len(s_list_model)) 
        #print("model list length: ",len(add_list_model_final)) 
        #print("all model evals length: ", len(all_model_eval))

        #if i%20 ==0: 
        #    cm.save_archive(self.model_archive, "model_gen_"+str(i), params, self.log_dir)
        #    print("Model gen: ", i)
        #    print("Model archive size: ", len(self.model_archive))
        print(f'Individuals evaluated on model: {len(s_list_model)}\nCurrent valid population at gen {gen}: {len(add_list_model_final)}')
        gen += 1
    self.model_eval_time = time.time() - start
    print(f"Random model emitter ended in {self.model_eval_time} after {gen} gen")
    return add_list_model_final, all_model_eval

    ## Return the archive
