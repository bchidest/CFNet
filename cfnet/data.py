import os
import random
import h5py
import numpy as np
import math
from PIL import Image
from scipy import ndimage
from scipy.stats import truncnorm


def stretch_images(images, min_values, max_values):
    perc = np.percentile(images, [0.1, 99.9], axis=[0, 1])
    min_values = perc[0, :]
    max_values = perc[1, :]
    images_stretched = images - min_values
    images_stretched[images_stretched < 0] = 0
    images_stretched = images_stretched / ((max_values - min_values).astype('float32'))
    images_stretched[images_stretched > 1] = 1
    return images_stretched


def add_jitter(image, offset, image_size):
    jittered_offset_x = random.randint(0, 2*offset)
    jittered_offset_y = random.randint(0, 2*offset)
    image_jittered = image[jittered_offset_x:image_size + jittered_offset_x,
                           jittered_offset_y:image_size + jittered_offset_y]
    return image_jittered


def crop_center(image, offset, image_size):
    offset = int(offset/2)
    image_cropped = image[offset:offset + image_size,
                          offset:offset + image_size]
    return image_cropped


class HDF5DataSet(object):

    def __init__(self, images, labels, label_list,
                 image_shape_full, jitter_offset=0, crop=False,
                 min_values=None, max_values=None,
                 calculate_min_max_values=False,
                 n_samples_for_min_max_estimate=-1,
                 balance_classes=True,
                 rotate_images=False,
                 flip_images=False,
                 stretch_images=True,
                 one_hot=True):
        # TODO: assumes labels are one-hot encoded! should change
        self._n_examples = len(images)
        # TODO: check that dtype is float
        self._image_dtype = images[0].dtype
        self._image_shape_full = image_shape_full
        self._jitter_offset = jitter_offset
        self._crop = crop
        self._rotate_images = rotate_images
        self._flip_images = flip_images
        self._stretch_images = stretch_images
#        self._images = images
        self._images = np.reshape(images, [self._n_examples] + image_shape_full, order='F')
        self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._index_in_no_repeats = 0
        self._label_list = label_list
        self._n_classes = len(label_list)
        self._balance_classes = balance_classes
        if len(self._image_shape_full) < 3:
            self._image_depth = 1
        else:
            self._image_depth = self._image_shape_full[2]
        self._image_size = image_shape_full[0] - 2*jitter_offset
        self._image_shape = [self._image_size, self._image_size, self._image_depth]
        # TODO: assumes both image_size and image_size full are even!
        self._image_size_offest = (image_shape_full[0] - self._image_size) / 2
        perm = np.arange(self._n_examples)
        self._ind = [list(perm[labels[:, i] == 1]) for i in range(self._n_classes)]
        self._n_samples_per_label = []
        for i in range(self._n_classes):
            self._n_samples_per_label.append(len(self._ind[i]))
        #self._ind = [[i for i in range(self._n_examples) if label_names[i] == l] for l in self._label_list]
        #self._n_samples_per_label = [len(ind_i) for ind_i in self._ind]
        self._ind_max_label = np.argmax(np.array(self._n_samples_per_label))
        self._max_n_samples = self._n_samples_per_label[self._ind_max_label]
        self._n_examples_in_epoch = self._n_examples
        if balance_classes:
            self._n_examples_in_epoch = self._max_n_samples*self._n_classes
            # How many times must all samples of under-represented classes
            # be resampled
            self._n_repeats = [self._max_n_samples / n for n in self._n_samples_per_label]
            # How many additional samples of under-represented classes must be
            # generated
            self._n_remainder = [self._max_n_samples % n for n in self._n_samples_per_label]
        self._one_hot = True
#        self._images_batch = np.zeros((self._n_examples_in_epoch,
#                                       self._image_size, self._image_size,
#                                       self._image_depth), dtype='uint8')
        self.shuffle_data()
        self._min_values = min_values
        self._max_values = max_values
        if calculate_min_max_values:
            self.calculate_percentiles(n_samples_for_min_max_estimate)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_examples(self):
        return self._n_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def calculate_percentiles(self, n_samples_for_min_max_estimate):
        if n_samples_for_min_max_estimate == -1:
            images_temp = np.reshape(self._images[:],
                                     [self._n_examples] + self._image_shape_full, order='F')
        else:
            ind = list(np.random.choice(self._n_examples, n_samples_for_min_max_estimate,
                                        replace=False))
            ind.sort()
            images_temp = np.reshape(self._images[ind], [n_samples_for_min_max_estimate] +
                                     self._image_shape_full, order='F')
        perc = np.percentile(images_temp, [0.1, 99.9], axis=[0, 1, 2])
        self._min_values = perc[0, :]
        self._max_values = perc[1, :]

    def shuffle_data(self):
        '''Shuffle the ordering of the samples.'''
        if self._balance_classes:
            self._perm = []
            for i in range(self._n_classes):
                self._perm += self._ind[i]*self._n_repeats[i]
                self._perm += random.sample(self._ind[i], self._n_remainder[i])
        else:
            self._perm = range(self._n_examples)
        if self._flip_images:
            self._flip_ver = np.random.random_integers(0, 1, self._n_examples_in_epoch).astype('bool')
            self._flip_hor = np.random.random_integers(0, 1, self._n_examples_in_epoch).astype('bool')
        else:
            self._flip_ver = np.zeros((self._n_examples_in_epoch,), dtype='bool')
            self._flip_hor = np.zeros((self._n_examples_in_epoch,), dtype='bool')
        self._rot = np.random.random_integers(0, 3, self._n_examples_in_epoch)
        random.shuffle(self._perm)

    def reset_epoch(self):
        '''Reset the epoch counter.'''
        self._index_in_epoch = np.inf

    def reset_no_repeats(self):
        '''Reset the no_repeats epoch counter.'''
        self._index_in_no_repeats = np.inf

    def next_batch(self, batch_size):
        '''Return the next batch.'''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # Finished epoch
        if self._index_in_epoch > self._n_examples_in_epoch:
            self.shuffle_data()
            self._epochs_completed += 1
            self._index_in_epoch = batch_size
            start = 0
        end = self._index_in_epoch
        batch_ind = self._perm[start:end]
        images = self._images[batch_ind]
        labels = self._labels[batch_ind]
        return images, labels

    def next_batch_hdf5(self, batch_size):
        '''Return the next batch.'''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # Finished epoch
        if self._index_in_epoch > self._n_examples_in_epoch:
            self.shuffle_data()
            self._epochs_completed += 1
            self._index_in_epoch = batch_size
            start = 0
        end = self._index_in_epoch
        batch_ind = self._perm[start:end]
        # indices cannot have repeats, so do them separately
        bi_unique = set()
        bi_repeats = []
        for bi in batch_ind:
            if bi in bi_unique:
                bi_repeats.append(bi)
            else:
                bi_unique.add(bi)
        if bi_repeats:
            batch_ind = list(set(batch_ind))
        n_repeats = len(bi_repeats)
        # indices must be in increasing order for hdf5 access
        batch_ind.sort()
        images_temp = np.reshape(self._images[batch_ind], [batch_size - n_repeats] + self._image_shape_full, order='F')
        for bi in bi_repeats:
            images_temp = np.concatenate([images_temp,
                                     np.reshape(self._images[bi], [1] + self._image_shape_full, order='F')],
                                    axis=0)
        labels = self._labels[batch_ind]
        for bi in bi_repeats:
            labels = np.concatenate([labels, np.expand_dims(self._labels[bi], axis=0)], axis=0)
#        images_temp = stretch_images(images_temp, self._min_values, self._max_values)
        images = np.zeros((batch_size, self._image_size, self._image_size,
                           self._image_depth), dtype='float32')
        if self._rotate_images:
            for i in range(batch_size):
#                images[i] = np.rot90(images[i], self._rot[i])
                ndimage.rotate(images_temp[i], random.randint(0, 360), reshape=False,
                               output=images_temp[i], mode='reflect')
        if self._stretch_images:
            for i in range(batch_size):
                images_temp[i] = stretch_images(images_temp[i], self._min_values, self._max_values)
        if self._jitter_offset != 0:
            if self._crop:
                for i in range(batch_size):
                    images[i] = crop_center(images_temp[i], self._jitter_offset,
                                            self._image_size)
            else:
                for i in range(batch_size):
                    images[i] = add_jitter(images_temp[i], self._jitter_offset,
                                           self._image_size)
        else:
            image_size_end = self._image_size + self._image_size_offest
            for i in range(batch_size):
                images[i] = images_temp[i][self._image_size_offest:
                                           image_size_end,
                                           self._image_size_offest:
                                           image_size_end, :]
        del images_temp
        for i in range(batch_size):
            if self._flip_hor[i]:
                images[i] = np.fliplr(images[i])
            if self._flip_ver[i]:
                images[i] = np.flipud(images[i])
        return images, labels

    def next_batch_no_repeats(self, batch_size):
        '''Return the next batch without repeats.'''
        start = self._index_in_no_repeats
        self._index_in_no_repeats += batch_size

        # Processed all unique samples
        if self._index_in_no_repeats > self._n_examples:
            self._index_in_no_repeats = batch_size
            start = 0
        end = self._index_in_no_repeats
        labels = self._labels[start:end]
        images = self._images[start:end]
        return images, labels

    def next_batch_no_repeats_hdf5(self, batch_size):
        '''Return the next batch without repeats.'''
        start = self._index_in_no_repeats
        self._index_in_no_repeats += batch_size

        # Processed all unique samples
        if self._index_in_no_repeats > self._n_examples:
            self._index_in_no_repeats = batch_size
            start = 0
        end = self._index_in_no_repeats
        images_temp = np.reshape(self._images[start:end], [batch_size] + self._image_shape_full, order='F')
        labels = self._labels[start:end]
#        images_temp = stretch_images(images_temp, self._min_values, self._max_values)
        images = np.zeros((batch_size, self._image_size, self._image_size,
                           self._image_depth), dtype='float32')
        if self._stretch_images:
            for i in range(batch_size):
                images_temp[i] = stretch_images(images_temp[i], self._min_values, self._max_values)
        image_size_end = self._image_size + self._image_size_offest
        for i in range(batch_size):
            images[i] = images_temp[i][self._image_size_offest:
                                       image_size_end,
                                       self._image_size_offest:
                                       image_size_end, :]

        return images, labels

    def save_example_images(self, output_dir, channel_order=[0, 1]):
        # TODO: this code is no longer correct! (10/30)
        for ind in self._ind:
            # Save as RGB
            img = np.zeros((self._image_size, self._image_size, 3), dtype='uint8')
            #img_temp = np.reshape(self._images[ind[0]] / self._intensity_norm_constant, self._image_shape, order='F')
            img_temp = np.reshape(self._images[ind[0]] / 100, self._image_shape_full, order='F')
            img_temp[img_temp > 1.0] = 1.0
            img_temp = (255*img_temp).astype('uint8')
            label = self._label_list[np.argmax(self._labels[ind[0]])]
            for i, c in zip(range(len(channel_order)), channel_order):
                img[:, :, i] = img_temp[:, :, c]
            Image.fromarray(img).save(os.path.join(output_dir, label + '.png'))

    def get_examples_of_classes(self):
        images = np.zeros((self._n_classes, self._image_size, self._image_size,
                           self._image_depth), dtype='float32')
        labels = range(self._n_classes)
        image_size_end = self._image_size + self._image_size_offest
        for c in range(self._n_classes):
            idx = random.choice(self._ind[c])
            images[c] = np.reshape(self._images[idx], [1] + self._image_shape_full, order='F')[0, self._image_size_offest:
                                       image_size_end,
                                       self._image_size_offest:
                                       image_size_end, :]
            if self._stretch_images:
                images[c] = stretch_images(images[c], self._min_values, self._max_values)

        return images, labels


def read_image_patch_data_set_hdf5(data_set_filename,
                                   images_key,
                                   labels_key,
                                   label_list_key,
                                   image_shape,
                                   jitter_offset=0,
                                   crop=False,
                                   min_values=None,
                                   max_values=None,
                                   calculate_min_max_values=False,
                                   rotate_images=False,
                                   flip_images=False,
                                   stretch_images=True,
                                   one_hot=True,
                                   balance_classes=True):
    data_set = h5py.File(data_set_filename)
    return HDF5DataSet(data_set[images_key], data_set[labels_key],
                       list(data_set.attrs[label_list_key]),
                       image_shape,
                       jitter_offset=jitter_offset,
                       crop=crop,
                       min_values=min_values,
                       max_values=max_values,
                       calculate_min_max_values=calculate_min_max_values,
                       rotate_images=rotate_images,
                       flip_images=flip_images,
                       stretch_images=stretch_images,
                       one_hot=one_hot,
                       balance_classes=balance_classes)


def subsample_hdf5_data_set(data_set_filename,
                            images_key,
                            labels_key,
                            n_samples,
                            output_filename=None):
    if not output_filename:
        output_filename = os.path.splitext(data_set_filename)[0] + \
                '_subsampled_' + str(n_samples) + '.hdf5'
    data_set = h5py.File(data_set_filename)
    labels = data_set[labels_key]
    n_examples = len(labels)
    n_classes = data_set[labels_key].shape[1]
    n_samples_per_label = [n_samples] * n_classes
    perm = np.arange(n_examples)
    ind = [perm[labels[:, i] == 1] for i in range(n_classes)]
    for i in range(n_classes):
        n = len(ind[i])
        if n < n_samples:
            n_samples_per_label[i] = n
    n_total_samples = sum(n_samples_per_label)
    data_dim = data_set[images_key].shape[1]
    data = np.zeros((n_total_samples, data_dim))
    labels = np.zeros((n_total_samples, n_classes), dtype='uint8')
    idx = 0
    for i in range(n_classes):
        ind_sub = random.sample(ind[i], n_samples_per_label[i])
        ind_sub.sort()
        print(n_samples_per_label[i], ind_sub)
        idx_new = idx + n_samples_per_label[i]
        data[idx:idx_new] = data_set[images_key][ind_sub]
        labels[idx:idx_new] = data_set[labels_key][ind_sub]
        idx = idx_new
    f_out = h5py.File(output_filename, 'w')
    f_out.create_dataset(images_key, data=data)
    f_out.create_dataset(labels_key, data=labels)
    for a in data_set.attrs.keys():
        f_out.attrs.create(a, data_set.attrs[a])
    f_out.close()
    data_set.close()


class GaussianGenerator(object):
    # NOTE: All angles are considered as radians.
    # TODO: Check that x and y are not swapped.
    def __init__(self, x_hidden, y_hidden, sigma_hidden, sigma, n_points,
                 n_points_var):
        self.x_hidden = x_hidden
        self.y_hidden = y_hidden
        self.sigma_hidden = sigma_hidden
        self.sigma = sigma
        self.n_points = n_points
        self.n_points_var = n_points_var

    @classmethod
    def random(cls, x_max=1, y_max=1, sigma_hidden=1.0,
               sigma_max=1.0, n_points=20, n_points_var=3):
        x_hidden = random.randint(-x_max, x_max)
        y_hidden = random.randint(-y_max, y_max)
        theta = 2*np.pi*random.random()
        sigma_hidden = np.array([(sigma_hidden, 0), (0, sigma_hidden)], dtype='float')
        sigma_x = math.sqrt(sigma_max)*random.random()
        sigma_y = math.sqrt(sigma_max)*random.random()
        sigma = np.array([(sigma_x, 0), (0, sigma_y)], dtype='float')
        rot_matrix = np.array([(math.cos(theta), -math.sin(theta)),
                               (math.sin(theta), math.cos(theta))])
        sigma = np.dot(rot_matrix, np.dot(sigma, np.dot(sigma, np.linalg.inv(rot_matrix))))
        return cls(x_hidden, y_hidden, sigma_hidden, sigma, n_points,
                   n_points_var)

    def generate_points(self, theta, int_locations=True):
        offset = np.random.multivariate_normal([self.x_hidden, self.y_hidden],
                                               self.sigma_hidden,
                                               size=1)
        x_mean = offset[0, 0]
        y_mean = offset[0, 1]
        r_mean = math.sqrt(x_mean**2 + y_mean**2)
        theta_mean = math.atan2(y_mean, x_mean) + theta
        x_mean = r_mean*math.cos(theta_mean)
        y_mean = r_mean*math.sin(theta_mean)
        rot_matrix = np.array([(math.cos(theta_mean), -math.sin(theta_mean)),
                               (math.sin(theta_mean), math.cos(theta_mean))])
        sigma = np.dot(rot_matrix, np.dot(self.sigma, np.linalg.inv(rot_matrix)))
        n_points = int(np.random.normal(self.n_points, self.n_points_var))
        points = np.random.multivariate_normal([x_mean, y_mean], sigma,
                                               size=n_points)
        if int_locations:
            points = points.astype('int')
        return points


class RandomImage(object):
    def __init__(self,
                 img_size=51,
                 n_gaussians=5,
                 n_points_per_gaussian=[20, 20],
                 n_points_var=3,
                 gaussian_mean_pct_max=0.7,
                 sigma_hidden_pct_max=0.1,
                 sigma_pct_max=0.3,
                 point_intensity_variance=20,
                 point_intensity_mean_variance=20,
                 noise_scale=7,
                 background_scale=7,
                 offset_max=0,
                 flips=True,
                 jitter_offset=3):
        self.img_size = img_size
        self.n_gaussians = n_gaussians
        self.n_points_per_gaussian = n_points_per_gaussian
        self.n_points_var = n_points_var
        self.point_intensity_variance = point_intensity_variance
        self.point_intensity_mean_variance = point_intensity_mean_variance
        self.noise_scale = noise_scale
        self.background_scale = background_scale
        self.offset_max = offset_max
        self.flips = flips
        self.jitter_offset = jitter_offset
        self.gaussians = []

        # Create GaussianGenerators
        self.img_radius = img_size / 2
        x_max = int(gaussian_mean_pct_max*self.img_radius)
        y_max = int(gaussian_mean_pct_max*self.img_radius)
        sigma_hidden = sigma_hidden_pct_max*img_size
        sigma_max = sigma_pct_max*img_size
        # Number of points per Gaussian chosen from Unif dist
        if n_points_per_gaussian[0] == n_points_per_gaussian[1]:
            n_points_per_gaussian[1] += 1
        n_points = np.random.randint(n_points_per_gaussian[0],
                                     n_points_per_gaussian[1], n_gaussians)
        for g in range(n_gaussians):
            self.gaussians.append(GaussianGenerator.random(x_max,
                                                           y_max,
                                                           sigma_hidden,
                                                           sigma_max,
                                                           n_points[g],
                                                           self.n_points_var))

    def generate_images(self, n_images=1, add_noise=True, save_dir='',
                        random_theta=True, flip_images=True):
        if random_theta:
            theta = 2*np.pi*np.random.uniform(size=n_images)
        else:
            theta = 2*np.pi*np.linspace(0, 1, n_images)
        img_size_buffered = 2*self.img_size - 1
        img_radius_buffered = img_size_buffered / 2
        img = np.zeros((n_images, img_size_buffered, img_size_buffered), dtype='uint16')

        for i in range(n_images):
            # Draw background value.
            background = int(np.random.exponential(scale=self.background_scale))
            img[i] += background
            # Draw mean point intensity.
            point_mean = int(np.random.normal(128, self.point_intensity_mean_variance))
            offset_x = random.randint(-self.offset_max, self.offset_max)
            offset_y = random.randint(-self.offset_max, self.offset_max)
            for g in self.gaussians:
                # Draw number of points for Gaussian.
                points = g.generate_points(theta[i])
                point_intensities = self.point_intensity_variance *\
                        np.random.uniform(size=len(points)) + point_mean
                for p in range(len(points)):
                    img[i, img_radius_buffered + points[p, 0] + offset_x,
                        img_radius_buffered + points[p, 1] + offset_y] = point_intensities[p]
            img[i] = ndimage.gaussian_filter(img[i], sigma=(1, 1))
            if self.flips and flip_images:
                if random.randint(0, 1):
                    img[i] = np.fliplr(img[i])
        # Add exponential noise.
        if add_noise:
            img += (np.random.exponential(scale=self.noise_scale,
                                          size=[n_images,
                                                img_size_buffered,
                                                img_size_buffered])).astype('uint8')
        # Threshold pixel intensities.
        img[img >= 255] = 255
        # Crop images to appropriate shape.
        img_final = np.zeros((n_images, self.img_size-1 + 2*self.jitter_offset, self.img_size-1 + 2*self.jitter_offset), dtype='uint8')
        jitter_offset_x = np.random.randint(-self.jitter_offset,
                                            self.jitter_offset + 1, n_images)
        jitter_offset_y = np.random.randint(-self.jitter_offset,
                                            self.jitter_offset + 1, n_images)
        for i in range(n_images):
            img_final[i] = img[i, self.img_radius - self.jitter_offset+1 + jitter_offset_x[i]:
                               -(self.img_radius - self.jitter_offset) + jitter_offset_x[i],
                               self.img_radius - self.jitter_offset+1 + jitter_offset_y[i]:
                               -(self.img_radius - self.jitter_offset) + jitter_offset_y[i]]
        img = img_final
        if save_dir != '':
            os.mkdir(save_dir)
            print('Saving images.')
            for i in range(n_images):
                Image.fromarray(img[i]).save(os.path.join(save_dir, 'img_' + str(i) + '.png'))
        if n_images == 1:
            img = img[0, :, :]
        return img


def generate_random_image_h5py(n_classes=50,
                               n_images_per_class=[200],
                               img_size=51,
                               n_gaussians=10,
                               n_points_per_gaussian=[20, 20],
                               flips=True,
                               jitter_offset=3,
                               save_dir='',
                               save_examples=True,
                               filename='random_images_'):
    n_rotations = 10
    n_datasets = len(n_images_per_class)
    rimg = []
    for k in range(n_classes):
        rimg.append(RandomImage(img_size=img_size,
                                n_gaussians=n_gaussians,
                                n_points_per_gaussian=n_points_per_gaussian,
                                flips=flips,
                                jitter_offset=jitter_offset))
    # Create filename number suffix
    filename_num = [str(n_images_per_class[i]) for i in range(n_datasets)]

    # Loop over number of datasets.
    for i in range(n_datasets):
        print("# images per class = " + str(n_images_per_class[i]))
        n_imgs = n_classes*n_images_per_class[i]
        imgs = np.zeros((n_imgs, img_size-1+2*jitter_offset, img_size-1+2*jitter_offset), dtype='uint8')
        labels_onehot = np.zeros((n_imgs, n_classes), dtype='uint8')
        for k in range(n_classes):
            imgs[k*n_images_per_class[i]:(k+1)*n_images_per_class[i]] =\
                rimg[k].generate_images(n_images_per_class[i])
            labels_onehot[k*n_images_per_class[i]:(k+1)*n_images_per_class[i], k] = 1
        imgs = np.reshape(imgs, [n_imgs, (img_size-1+2*jitter_offset)**2], order='F')
        # Create HDF5 file.

        imgs_h5py = h5py.File(os.path.join(save_dir,
                                           filename + filename_num[i] + '.h5py'), 'w')
        attrs = []
        for k in range(n_classes):
            attrs.append('class' + str(k))
        attrs = np.array(attrs)
        imgs_h5py.create_dataset('data', data=imgs)
        imgs_h5py.create_dataset('index', data=labels_onehot)
        imgs_h5py.attrs.create('index_columns', data=attrs)
        imgs_h5py.close()
    if save_examples:
        for k in range(n_classes):
            out_dir = os.path.join(save_dir, 'random_image_examples')
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            img_name = os.path.join(out_dir, 'class' + str(k) + '.png')
            Image.fromarray(np.reshape(imgs[k*n_images_per_class[n_datasets-1]], [img_size-1+2*jitter_offset, img_size-1+2*jitter_offset], order='F')).save(img_name)
        out_sub_dir = os.path.join(out_dir, 'rotation_example')
        if not os.path.isdir(out_sub_dir):
            os.makedirs(out_sub_dir)
        imgs_rotations = rimg[0].generate_images(n_rotations,
                                                 random_theta=False,
                                                 flip_images=False)
        for r in range(n_rotations):
            img_name = os.path.join(out_sub_dir, 'class0_rotation' + str(r) + '.png')
            Image.fromarray(imgs_rotations[r]).save(img_name)
