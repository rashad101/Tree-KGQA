wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip
unzip wiki.simple.zip
mv wiki.simple.bin data/
rm wiki.simple.vec wiki.simple.zip

wget http://dl.fbaipublicfiles.com/elq/entity.jsonl
mv entity.jsonl data/
mkdir outputs/EL
mkdir outputs/ablation
mkdir outputs/kgqa