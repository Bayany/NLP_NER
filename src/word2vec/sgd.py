#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000
import os.path as op
import pickle
import random
import numpy as np
import logging
import os
import glob 

def load_saved_params(label):
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob(f"models/{label}.word2vec_*.pickle"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[1])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = f"models/{label}.word2vec.npy" 
        state_file =  f"models/{label}.word2vec_{st}.pickle" 
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params,label):
    params_file = f"models/{label}.word2vec.npy"
    np.save(params_file, params)
    with open( f"models/{label}.word2vec_{iter}.pickle", "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10,label="all_chapters"):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved and os.path.exists(f"models/{label}.word2vec.npy"):
        start_iter, oldx, state = load_saved_params(label)
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    logging.info(f"Training word2vec ({label})")
    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        ### YOUR CODE HERE (~2 lines)
        loss, grad = f(x)
        x -=  step * grad
        ### END YOUR CODE

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            logging.info("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x, label)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
