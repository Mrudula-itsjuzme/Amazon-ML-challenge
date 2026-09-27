#!/bin/bash
while true; do
  if grep -q "Done!" /home/mrudula/.gemini/antigravity/brain/69998e71-d780-4451-9d0c-885985741a2c/.system_generated/tasks/task-546.log; then
    echo "Task finished! Zipping..."
    zip -r mrudula_submission.zip src output README.md Documentation_template.md
    break
  fi
  sleep 2
done
