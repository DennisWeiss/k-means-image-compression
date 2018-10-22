import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.cluster


def flatten_img(img):
    pixel_lst = []
    for row in img:
        for pixel in row:
            pixel_lst.append(pixel)
    return pixel_lst


def compute_squared_euclidean_distance(a, b):
    assert len(a) == len(b)
    squared_distance = 0
    for i in range(len(a)):
        squared_distance += (a[i] - b[i])**2
    return squared_distance


def get_closest_cluster_center(pixel, cluster_centers):
    closest_cluster_center = None
    lowest_squared_distance = float('inf')
    for cluster_center in cluster_centers:
        squared_euclidean_distance = compute_squared_euclidean_distance(pixel, cluster_center)
        if squared_euclidean_distance < lowest_squared_distance:
            lowest_squared_distance = squared_euclidean_distance
            closest_cluster_center = cluster_center
    return closest_cluster_center


def apply_k_means_result(img, k_means_result):
    cluster_centers = k_means_result.cluster_centers_
    new_img = []
    for row in img:
        new_row = []
        for pixel in row:
            new_row.append(get_closest_cluster_center(pixel, cluster_centers))
        new_img.append(new_row)
    return new_img


# image_path = input('Enter image path:')
image_path = 'new_york.png'

img = mpimg.imread(image_path)

k_means_result = sklearn.cluster.KMeans(n_clusters=10, max_iter=100).fit(flatten_img(img))

new_img = apply_k_means_result(img, k_means_result)
imgplot = plt.imshow(new_img)


plt.imsave('k_means_{}'.format(image_path), new_img)
plt.show()
