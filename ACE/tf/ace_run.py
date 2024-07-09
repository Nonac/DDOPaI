"""This script runs the whole ACE method."""


import sys
import os
import numpy as np
import sklearn.metrics as metrics
from tcav import utils
import tensorflow as tf

from ACE.tf import ace_helpers
from ACE.tf.ace import ConceptDiscovery
from _PATH import *

def main(args,model_path):

  ###### related DIRs on CNS to store results #######
  discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
  results_dir = os.path.join(args.working_dir, 'results/')
  cavs_dir = os.path.join(args.working_dir, 'cavs/')
  activations_dir = os.path.join(args.working_dir, 'acts/')
  results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
  if tf.io.gfile.exists(args.working_dir):
    tf.compat.v1.gfile.DeleteRecursively(args.working_dir)
  tf.io.gfile.makedirs(args.working_dir)
  tf.io.gfile.makedirs(discovered_concepts_dir)
  tf.io.gfile.makedirs(results_dir)
  tf.io.gfile.makedirs(cavs_dir)
  tf.io.gfile.makedirs(activations_dir)
  tf.io.gfile.makedirs(results_summaries_dir)
  random_concept = 'random_discovery'  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = ace_helpers.make_model(
      sess, args.teacher_model,model_path, args.labels_path)
  # Creating the ConceptDiscovery class instance

  cd = ConceptDiscovery(
      mymodel,
      args.target_class,
      random_concept,
      args.bottlenecks.split(','),
      sess,
      #args.source_dir,
      RANDOM_PATH,
      activations_dir,
      cavs_dir,
      num_random_exp=args.num_random_exp,
      channel_mean=True,
      max_imgs=args.max_imgs,
      min_imgs=args.min_imgs,
      num_discovery_imgs=args.max_imgs,
      num_workers=args.num_parallel_workers)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': [15,50,80]})
  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images')
  tf.io.gfile.makedirs(image_dir)
  ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  ace_helpers.save_concepts(cd, discovered_concepts_dir)
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  scores = cd.tcavs(test=False)
  ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir + 'ace_results.txt')
  # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
  # Delete concepts that don't pass statistical testing
  cd.test_and_remove_concepts(scores)
  sess.close()




