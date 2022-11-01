wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip
unzip wiki.simple.zip
mv wiki.simple.bin data/
rm wiki.simple.vec wiki.simple.zip

wget http://dl.fbaipublicfiles.com/elq/entity.jsonl
mv entity.jsonl data/
mkdir outputs/EL
mkdir outputs/ablation
mkdir outputs/kgqa
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2
bzip2 -d enwiki_20180420_300d.txt.bz2
mv enwiki_20180420_300d.txt data/