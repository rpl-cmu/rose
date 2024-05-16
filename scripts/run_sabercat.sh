
for bag in /mnt/bags/best/*; do
    echo "\033[0;31m ${bag}\033[0m"
    python scripts/2jrl_run.py --sabercat ${bag}
done

curl -d "Finished running sabercat data" ntfy.sh/easton_work