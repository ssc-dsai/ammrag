build.ps1 -stage 1
wsl --exec "cd /mnt/c/Users/$(whoami)/go/src/github.com/cilium/cc2go &&
build.ps1 -stage 2
