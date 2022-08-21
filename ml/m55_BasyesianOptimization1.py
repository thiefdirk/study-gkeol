param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4)}

def y_function(x1, x2):
    return -x1**2 - (x2-2)**2 + 10

from bayes_opt import BayesianOptimization # pip install bayesian-optimization

optimizer = BayesianOptimization(f=y_function, 
                                 pbounds=param_bounds, 
                                 random_state=1234)

optimizer.maximize(init_points=2, n_iter=20, n_jobs=-1) # n_jobs=-1 : cpu를 모두 사용하겠다
# maximize : 최대값을 찾는다.
# 2번의 랜덤값을 찾고, 3번의 최적값을 찾는다.

print(optimizer.max) # 최대값을 찾는다.