# How GPS Positioning Works

The Global Positioning System determines a receiver's location by measuring its distance from several satellites orbiting the Earth. Each satellite continuously broadcasts a signal containing its precise orbital position and the exact time the signal was transmitted, stamped by an onboard atomic clock. A GPS receiver compares the arrival time of signals from multiple satellites with their transmission times.

Because the signals travel at the speed of light, each time delay converts directly into a distance. Knowing the distance from one satellite places the receiver somewhere on a sphere centered on that satellite. Intersecting the spheres from three satellites narrows the position to at most two points, and a fourth satellite resolves the ambiguity while also correcting the receiver's inexpensive clock, which cannot keep time as precisely as the satellites' atomic clocks. This geometric method is called trilateration.

Accuracy depends on several error sources. Signals slow slightly as they pass through the ionosphere and troposphere, buildings and cliffs reflect signals and create multipath errors, and satellite geometry matters: satellites spread widely across the sky yield a more precise fix than satellites clustered together, an effect called dilution of precision.

Modern receivers routinely achieve accuracy within a few meters, and differential techniques that compare against a ground station at a known location can push accuracy to centimeters. The satellite constellation is maintained with at least twenty-four operational satellites so that at least four are visible from almost any point on Earth at any time.
