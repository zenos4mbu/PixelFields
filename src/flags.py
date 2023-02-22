from absl import flags

flags.DEFINE_string('dataset', 'SINGLE', 'Dataset name.')
flags.DEFINE_string('exp_name', 'SINGLE_sweep', 'Experiment name.')

flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('grid_type', "NGLOD", 'type of grid.')
flags.DEFINE_string('image_path', "./input/lena.jpg", 'Image path.')
flags.DEFINE_integer('num_LOD', 3, 'Levels of detail.')
flags.DEFINE_integer('image_size', 200, 'Input images size.')
flags.DEFINE_integer('n_channels', 3, 'Number of image channels.')
flags.DEFINE_integer('trainset_size', 100, 'Size of the training set.')
flags.DEFINE_integer('feat_dim', 5, ' feat dimension in the grid')
flags.DEFINE_integer('base_lod', 2, ' base level of detail')
flags.DEFINE_integer('band_width', 1, 'codebook size')
flags.DEFINE_integer('hidden_dim', 256, 'Neural Network Hidden dim')
flags.DEFINE_float('feature_std', 0.01, 'Feature standard deviation')
flags.DEFINE_integer('min_grid_res', 8, 'Minimum grid resolution')
flags.DEFINE_integer('max_grid_res', 64, 'Maximum grid resolution')


flags.DEFINE_string('mode', 'train', 'train/test/demo.')
flags.DEFINE_integer('mapping_size', 2048, 'Dimension of the input mapping.')
flags.DEFINE_integer('mapping_multiplier', 50, 'Multiplier of the input mapping.')

flags.DEFINE_string('activation', 'RELU', 'RELU/SIN.')

flags.DEFINE_boolean('display', False,
                     'Display images during training.')

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_boolean('resume_training', False,
                     'Resume training using a checkpoint.')
flags.DEFINE_boolean('use_grid', False,
                     'Use grid of features')
flags.DEFINE_string('load_checkpoint_dir', '<path>',
                    'Load previous existing checkpoint.')
flags.DEFINE_integer('seed', 42, 'Seed.')
flags.DEFINE_integer('max_epochs', 50, 'Number of training epochs.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay.')
flags.DEFINE_float(
    'patience', 10, 'Number of epochs with no improvement after which learning rate will be reduced.')
flags.DEFINE_integer('num_workers', 8, 'Number of workers.')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus.')

flags.DEFINE_integer('accumulation', 1, 'Gradient accumulation iterations.')
