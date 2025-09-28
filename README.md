Init repo for rl-search-agents

*Requirements*

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