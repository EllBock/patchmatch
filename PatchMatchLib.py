import numpy as np


def patch_match(image, reference, patch_size=8, iterations=5, alpha=0.5):

    # Initialization
    a = np.array(image.convert('RGB'))
    b = np.array(reference.convert('RGB'))
    p = int(patch_size/2)

    a_h, a_w, a_d = np.size(a, 0), np.size(a, 1), np.size(a, 2)
    b_h, b_w, b_d = np.size(b, 0), np.size(b, 1), np.size(b, 2)
    pnum_a = a_h * a_w
    a_pad = np.zeros((a_h + p * 2, a_w + p * 2, a_d), dtype=np.uint8)
    a_pad[p:-p, p:-p] = a
    b_pad = np.zeros((b_h + p * 2, b_w + p * 2, b_d), dtype=np.uint8)
    b_pad[p:-p, p:-p] = b

    a_index = np.zeros((pnum_a, 2), dtype=np.int64)
    a_index[:, 0] = np.arange(pnum_a) / a_w
    a_index[:, 1] = np.arange(pnum_a) % a_w
    a_index = a_index.reshape(a_h, a_w, 2)

    # Random map
    mapping = (np.concatenate(
            (np.random.randint(0, high=b_h, size=(pnum_a, 1)),
             np.random.randint(0, high=b_w, size=(pnum_a, 1))), axis=1)
                .reshape(a_h, a_w, 2))
    offsets = mapping - a_index
    dist = np.zeros((a_h, a_w), dtype=np.float)

    for x in range(a_h):
        for y in range(a_w):
            dist[x, y] = distance(
                            (x, y),
                            a_pad, (mapping[x, y, 0], mapping[x, y, 1]),
                            b_pad, patch_size)

    # Execution
    for i in range(1, iterations+1):
        for p in range(a_h):
            for q in range(a_w):

                # Propagation
                if i % 2:  # odd
                    x = p
                    y = q
                    point1_a = (max(0, x-1), y)
                    point2_a = (x, max(0, y-1))
                else:  # even
                    x = a_h - p - 1
                    y = a_w - q - 1
                    point1_a = (min(x+1, a_h-1), y)
                    point2_a = (x, min(y+1, a_w-1))

                point1_b = np.array([x, y]) + offsets[point1_a[0], point1_a[1]]
                point2_b = np.array([x, y]) + offsets[point2_a[0], point2_a[1]]
                d_choice = np.array([dist[x, y]])
                o_choice = np.expand_dims(offsets[x, y], 0)

                if (not out_of_bounds(point1_b, (b_h, b_w))
                        and not np.array_equal(point1_b, mapping[x, y])):
                    d_choice = np.append(d_choice,
                                         distance((x, y), a_pad, point1_b,
                                                  b_pad, patch_size))
                    o_choice = np.append(o_choice,
                                         np.expand_dims((point1_b
                                                         - np.array([x, y])),
                                                        0), axis=0)

                if (not out_of_bounds(point2_b, (b_h, b_w))
                        and not np.array_equal(point2_b, mapping[x, y])):
                    d_choice = np.append(d_choice,
                                         distance((x, y), a_pad, point2_b,
                                                  b_pad, patch_size))
                    o_choice = np.append(o_choice,
                                         np.expand_dims((point2_b
                                                         - np.array([x, y])),
                                                        0), axis=0)

                # Random search
                k = 1
                point_b = np.array([x, y]) + o_choice[np.argmin(d_choice)]
                search_h, search_w = b_h * (alpha ** k), b_w * (alpha ** k)
                while search_h > 1 and search_w > 1:
                    search_min_r = int(max(point_b[0] - search_h, 0))
                    search_max_r = int(min(point_b[0] + search_h, b_h))
                    search_min_c = int(max(point_b[1] - search_w, 0))
                    search_max_c = int(min(point_b[1] + search_w, b_w))
                    rand_b = np.array(
                                [np.random.randint(search_min_r,
                                                   search_max_r),
                                 np.random.randint(search_min_c,
                                                   search_max_c)])
                    d_choice = np.append(d_choice,
                                         distance((x, y), a_pad, rand_b,
                                                  b_pad, patch_size))
                    o_choice = np.append(o_choice,
                                         np.expand_dims((rand_b
                                                         - np.array([x, y])),
                                                        0), axis=0)
                    k += 1
                    search_h, search_w = b_h * (alpha ** k), b_w * (alpha ** k)

                choice = np.argmin(d_choice)
                dist[x, y] = d_choice[choice]
                offsets[x, y] = o_choice[choice]

    mapping = offsets + a_index

    return offsets, mapping


def distance(point_a, a, point_b, b, patch_size):
    p = int(patch_size / 2)
    patch_a = a[point_a[0]:point_a[0]+p*2,
                point_a[1]:point_a[1]+p*2].reshape(-1)/255
    patch_b = b[point_b[0]:point_b[0]+p*2,
                point_b[1]:point_b[1]+p*2].reshape(-1)/255
    return np.linalg.norm(patch_b - patch_a)**2


def out_of_bounds(point, shape):
    return (point[0] < 0 or point[0] >= shape[0]
            or point[1] < 0 or point[1] >= shape[1])
