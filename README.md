README
======


Build instructions
==================

1. Create a build directory in the project root folder
2. Run cmake inside the build directory.
3. Run make.


Command line interface of bin/run
=================================

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


Commands (results are sensitive to gamma)
=========================================

./bin/run -i ../images/motocycle0.png -j ../images/motocycle1.png -b 150 -t 45
./bin/run -i ../images/room0.jpg -j ../images/room1.jpg -b 150 -t 66
./bin/run -i ../images/umbrella0.jpg -j ../images/umbrella1.jpg -b 50 -t 43
./bin/run -i ../images/desk0.jpg -j ../images/desk1.jpg -b 120 -t 40


Contacts
========

Philipp Herrle: philipp.herrle@tum.de 
Utkarsh Pathak: utk.tum@gmail.com
Mingyip Cheung: mingyip.cheung@tum.de

If there are issues, please feel free to contact us.

References
========
[1] h. Pock, T. Schoenemann, H. Bischof, D. Cremers. A Convex Formulation of Continuous Multi-label Problems, Marseille, France 2008 
[2] D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, and P. Westling. High-resolution stereo datasets with subpixel-accurate ground truth.
In German Conference on Pattern Recognition (GCPR 2014), Münster, Germany, September 2014. 
