counter=0; cat temp | while read LINE; 
do ((counter++));
    array=( $LINE )
    echo $LINE > "/home/pfb16181/Desktop/tempok/files/${array[0]}"; 
done 