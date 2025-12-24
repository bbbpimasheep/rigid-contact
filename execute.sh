rm -r sim
mkdir sim
cp build/rigid_sim sim/rigid_sim
cp -r assets/ sim/assets/
cd sim
mkdir output
./rigid_sim