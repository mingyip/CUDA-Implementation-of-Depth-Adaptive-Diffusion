Depth Adaptive Diffusion
==================
<p align="center">
  <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/motocycle_blurred.jpg" width="600px">
</p>
We implemented a GPU version of paper "A Convex Formulation of Continuous Multi-label Problems". The program takes two stereo images where the position of camera are shifted right in the second image. With some optimization steps, we can calculate the corresponding pixel disparity of two images. The disparity map is an implicit representation of the depth distance between objects and the camera. Lastly, we can achieve depth adaptive diffusion image by blurring objects(pixels) in the background. 

Build instructions
==================

1. Create a build directory in the project root folder
2. Run cmake inside the build directory.
3. Run make.

Results
==================


 <table style="width:100%">
  <tr>
    <th> Original Image </th>
    <th> Background Blurred Image </th>
    <th> Disparity Map during updates </th>
  </tr>
  <tr>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/motocycle0.png" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/motocycle_blurred.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/motocycle_disparity.gif" width="270px">
    </th>
  </tr>
  <tr>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/room0.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/room_blurred.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/room_disparity.gif" width="270px">
    </th>
  </tr>
  <tr>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/umbrella0.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/umbrella_blurred.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/umbrella_disparity.gif" width="270px">
    </th>
  </tr>
  <tr>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/desk0.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/desk_blurred.jpg" width="270px">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/mingyip/GPU-Implementation-of-A-Convex-Formulation-of-Continuous-Multi-label-Problems-/master/images/desk_disparity.gif" width="270px">
    </th>
  </tr>
</table> 
<p> Special Thanks to Middlebury Stereo Datasets [2] </p>


Command line interface of bin/run
=================================
```
-i : left input image
-j : right input image
-n : max. number of iterations (termination criterion)
-p : energy change per pixel threshold to stop iterations (termination criterion)
-e : number of iterations after which energy calculation is done
-l : lambda
-b : number of gamma (layers)
-g : delta gamma
-m : number of iterations for diffusion
-t : foreground disparity threshold (higher disparities will not get blurred)
```

Example Commands (results are sensitive to gamma)
=========================================
```
./bin/run -i ../images/motocycle0.png -j ../images/motocycle1.png -b 150 -t 45
./bin/run -i ../images/room0.jpg -j ../images/room1.jpg -b 150 -t 66
./bin/run -i ../images/umbrella0.jpg -j ../images/umbrella1.jpg -b 50 -t 43
./bin/run -i ../images/desk0.jpg -j ../images/desk1.jpg -b 120 -t 40
```

Contacts
========

<p> Mingyip Cheung: mingyip.cheung@tum.de </p>
<p> Philipp Herrle: philipp.herrle@tum.de  </p>
<p> Utkarsh Pathak: utk.tum@gmail.com </p>
<p> Björn Häfner: bjoern.haefner@in.tum.de </p>

If there are issues, please feel free to contact us.

References
========
<p> [1] h. Pock, T. Schoenemann, H. Bischof, D. Cremers. A Convex Formulation of Continuous Multi-label Problems, Marseille, France 2008 </p>
<p> [2] D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, and P. Westling. High-resolution stereo datasets with subpixel-accurate ground truth.
In German Conference on Pattern Recognition (GCPR 2014), Münster, Germany, September 2014. </p>
