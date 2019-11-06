################################
litk - Lenticular Image Tool Kit
################################

:Authors: John Kay, Gilberto Galvis
:Email: john.kays2020@gmail.com, galvisgilberto@gmail.com
:Version: $revision: 0.1.1 $

This tool provides functions to work with lenticular images. So far, we have only developed functions to solve the problem backwards. That is, starting from the lenticular image (multi-view image), recover the original n images (the n views) that generated it.

Installation
------------

- Clone this repository on your machine either using http or ssh

Requirements
------------

- Python3: You need to have python installed on your machine

- Pillow library

Usage
-----

We can use this tool for two main cases: 1) recover and reconstruct a set of views from a mutivist iamgen and 2) make the animation with the previously recovered views

Recover and Reconstruct a set of views
======================================

For this mode of use we use the multiview2view tool. To execute it we must run

.. code:: shell

	python multiview2view.py <visual.json> <path-to-multiview-image> <number-of-views> <output-folder> <only-recovery>

As we see, each of the input arguments is passed via terminal on the same command line. For more information about the arguments, feel free to review the file ``multiview2view.py``

Example
+++++++

This repository contains a multi-view image with which we can test the tool. Below we show the respective example

.. code:: shell

	python multiview2views.py visual.json inputs/test1.bmp 37 views False

The result of this example is in the views folder

make the animation
==================

Additionally we can make an animation with the views obtained in the previous use mode. To execute it we must run

.. code:: shell

	python make_animation.py <view-folder> <video-name>

As we see, each of the input arguments is passed via terminal on the same command line. For more information about the arguments, feel free to review the file ``make_animation.py``

Example
+++++++

This repository contains a view folder with views recovery included with which we can test this tool. Below we show the respective example

.. code:: shell

	python make_animation.py views video_animation.avi

The result of this example is ``video_animation.avi``. Please feel free to open it.