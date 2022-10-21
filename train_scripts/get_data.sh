cd ..
git clone https://huggingface.co/datasets/sentence-transformers/parallel-sentences
sudo apt-get install git-lfs
cd parallel-sentences
git lfs pull
cd WikiMatrix
for x in `ls *.gz`; do
	gzip -d $x
done

