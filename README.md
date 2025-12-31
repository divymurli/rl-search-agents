# About
This repo contains the experiments for training neural ranking models using both contrastive as well as policy gradient (upcoming) training methods. Due to resource constraints, we restrict to training *second stage rerankers*, specifically aiming to be able to rerank within the top 1000 BM25-relevant documents (supervised pretraining) and within the top 10-100 reranked documents (policy gradient training).

We use pyserini as our vector datastore to be able to efficiently retrieve documents. Pyserini integrates core faiss capabilty for quick vector retrieval.

# Getting started

- Create a virtual environment, either with python `venv` or with `conda`. (We assume the user knows how to do this.)
- Run `pip install -r requirements.txt`

**Getting Java runtime properly configured for pyserini to work**
Tricky bits, especially with respect to getting `pyserini` working. These are instructions for getting pyserini to work on a Linux VM (where most GPU boxes run anyway):

```
# See what you have now
java -version
```

Then do the following:

```
# Install a modern JDK (Temurin 21 is fine; 17 also works)
sudo apt-get update
sudo apt-get install -y wget gnupg software-properties-common
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | sudo apt-key add -
sudo add-apt-repository --yes https://packages.adoptium.net/artifactory/deb
sudo apt-get update
sudo apt-get install -y temurin-21-jdk
```

Also install OpenJDK 21:

```
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk
```

Then point your system to their installation folders:

```
sudo update-alternatives --install /usr/bin/java  java  /usr/lib/jvm/temurin-21-jdk/bin/java  200

sudo update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/temurin-21-jdk/bin/javac 200
```

After this, update the configs and choose the option for `OpenJDK` for each of the belo commands (using your number keys/pad):

```
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

Confirm that your Java path is pointing to the right spot (should contain `OpenJDK 21` or similar):

```
which java
java -version
java --list-modules | grep jdk.incubator.vector
```

Once all's said and done, add the following two env vars to your `.bashrc`:

```
# make it persistent
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.bashrc
```

Open a new terminal tab (or a new tmux tab if ssh'd in); `which java` and `java --version` should return the following:

```
(pytorch) ubuntu@ip-172-31-32-84:~$ java --version
openjdk 21.0.8 2025-07-15
OpenJDK Runtime Environment (build 21.0.8+9-Ubuntu-0ubuntu122.04.1)
OpenJDK 64-Bit Server VM (build 21.0.8+9-Ubuntu-0ubuntu122.04.1, mixed mode, sharing)
```

# Generating the relevant data

To generate training data, run

```
python build_msmarco_beir_candidates.py \
  --irds-key irds/beir_msmarco_train \
  --index-mode prebuilt --index msmarco-v1-passage \
  --out data/candidates_train_1k_rm3_backoff.jsonl \
  --skip-log logs/skipped_train.jsonl \
  --topk 5000 --n-negs 999 --hard-cut 400 --mid-cut 1000 \
  --batch-size 256 --threads 16 --rm3-backoff
```

To generate the test data (from the official MSMarco dev set, which has 6,980 queries), run (*)

```
python build_msmarco_beir_candidates.py \       
  --irds-key irds/beir_msmarco_dev \  
  --index-mode prebuilt --index msmarco-v1-passage \
  --out data/candidates_dev_1k_rm3_backoff.jsonl \                         
  --skip-log logs/skipped_dev.jsonl \
  --topk 5000 --n-negs 999 --hard-cut 400 --mid-cut 1000 \
  --rm3-backoff --batch-size 256 --threads 16
```

To create a train-val split (1.5% is dev) to eval while training, run

```
python split_jsonl_by_hash.py \                    
  --inp data/candidates_train_1k_rm3_corrected_again.jsonl \
  --out-train training_data/train.jsonl \
  --out-dev training_data/dev.jsonl \
  --dev-rate 0.015
```

Note that by default, for time/cost saving reasons, we evaluate only the top 1000 queries on both the train-val set as well as the final test set.

# Training

Example training runs are provided in `run_experiments.sh`. To run a single training experiment, do e.g. (e.g. for grouped-k infoNCE)

```
python supervised/train_msmarco_supervised_infonce_updated.py  
--train_jsonl /path/to/training_data/train.jsonl  
--dev_jsonl  /path/to/training_data/dev.jsonl  
--loss_mode infonce_grouped 
--outdir ckpts/infonce_group_curriculum 
--batch_size 16 
--group_k 8 
--accum_steps 4
```

See `run_experiments.sh` for a comprehensive list of command line inputs.

# Evaluation

See `run_evals.sh` for a comprehensive list of evals to run. For a single experiment, update `DEFAULT_CKPT` and `DEFAULT_JSONL` to point respectively to the desired model checkpoint (from a training run) and the (test) set generated from above from (*).





