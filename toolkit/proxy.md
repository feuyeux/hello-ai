# http proxy

<https://x.com/getlantern_CN>

```sh
sudo apt install -y libayatana-appindicator3-1
sudo dpkg -i /mnt/c/Users/feuye/Downloads/lantern-installer-64-bit.deb

sudo vim /etc/profile.d/proxy.sh

export http_proxy="http://10.10.1.10:8080/"
export https_proxy="http://10.10.1.10:8080/"
```
