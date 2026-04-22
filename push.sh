#!/bin/bash

until git push; do
    echo "推送失败，可能是网络问题，5 秒后重试..."
    sleep 5
done

echo "推送成功！脚本结束。"