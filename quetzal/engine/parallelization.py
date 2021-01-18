from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def map_kwargs(func, iterables=[], show_progress=False, **kwargs):
    return [
        func(arg, **kwargs)
        for arg in (tqdm(iterables) if show_progress else iterables)
    ]


def parallel_map_kwargs(func, iterables, workers=1, **kwargs):
    if workers == 1:
        return map_kwargs(func, iterables, **kwargs)

    chunk_size = len(iterables) // workers
    if len(iterables) % workers > 0:
        chunk_size += 1

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(workers):
            results.append(
                executor.submit(
                    map_kwargs,
                    func,
                    iterables[i*chunk_size:(i+1)*chunk_size], 
                    **kwargs
                )
            )

    to_return = []
    for i in range(len(results)):
        r = results.pop(0)
        to_return += r.result()
        del r
    return to_return
