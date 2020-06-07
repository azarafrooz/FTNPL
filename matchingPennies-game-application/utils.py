def queue_update(queue, m, K, t, ft, inc=1):
    """
    Speciall queuing update for FTRL and FTNPL.
    :param queue: Queue containing the past params
    :param m: spacing parameter
    :param K: Queue size
    :param inc: hyper-parameter that needs to be adjusted
    :param t: current step
    :param ft: new model
    :return: updated queue and new spacing parameter
    """
    if t % m ==0 and len(queue)==K:
        queue.append(ft)  # we remove a model from the end of the queue, which is the oldest one
        return queue, m+inc
    elif len(queue) ==K: # override the head (first) element with the current model
        queue.pop()
        queue.append(ft)
        return queue, m
    else:
        queue.append(ft)
    return queue, m