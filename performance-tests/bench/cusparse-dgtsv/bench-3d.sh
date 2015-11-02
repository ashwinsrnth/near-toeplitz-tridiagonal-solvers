make run

printf "%15s \t %15s \t %15s \n" "N" "nrhs" "Time to solve (ms)"
for size in 32 64 128 256 512
do 
    nrhs=$(($size*$size))
    time=$(./run $size $nrhs)
    printf "%15d \t %15d \t %15.3f \n" "$size" "$nrhs" "$time"
done
