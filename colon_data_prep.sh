mkdir data/collect;

find data/* -type f -name '*.img' -print0 | 
while IFS= read -r -d '' file; do
    mv "$file" data/collect/ ;
done;

find data/* -type f -name '*.hdr' -print0 | 
while IFS= read -r -d '' file; do
    mv "$file" data/collect/ ;
done