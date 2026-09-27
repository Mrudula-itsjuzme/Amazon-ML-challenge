#!/bin/bash
set -e

# ==============================================================================
# AWS EC2 ML Acceleration Script
# ==============================================================================
# BEFORE RUNNING:
# You must request a Service Quota increase for EC2 "Running On-Demand Standard 
# (A, C, D, H, I, M, R, T, Z) instances" in ap-south-1. Request at least 64 vCPUs.
# ==============================================================================

INSTANCE_TYPE="r6i.xlarge"  # 64 vCPUs - Adjust based on your approved quota!
REGION="ap-south-1"
KEY_NAME="ml-challenge-key-$(date +%s)"
SEC_GROUP_NAME="ml-challenge-sg-$(date +%s)"
PROFILE="Mrudula"

echo "Fetching latest Amazon Linux 2023 AMI in $REGION..."
AMI_ID=$(aws ssm get-parameters --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64 --region $REGION --profile $PROFILE --query 'Parameters[0].Value' --output text)

# 1. Create Key Pair
echo "Creating SSH Key Pair..."
aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text --region $REGION --profile $PROFILE > ${KEY_NAME}.pem
chmod 400 ${KEY_NAME}.pem

# 2. Create Security Group
echo "Creating Security Group..."
SG_ID=$(aws ec2 create-security-group --group-name $SEC_GROUP_NAME --description "Allow SSH" --region $REGION --profile $PROFILE --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $REGION --profile $PROFILE > /dev/null

# 3. Launch Instance
echo "Launching $INSTANCE_TYPE instance (this requires the quota increase)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --region $REGION \
    --profile $PROFILE \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Waiting for instance $INSTANCE_ID to boot..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION --profile $PROFILE

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --profile $PROFILE \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance is running at $PUBLIC_IP"
echo "Waiting 45 seconds for SSH service to start..."
sleep 45

# 4. Sync files
echo "Syncing project to instance (transferring data, this may take a few minutes)..."
rsync -Pav -e "ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no" --exclude '.git' --exclude '__pycache__' ./ ec2-user@${PUBLIC_IP}:~/ml-challenge/

# 5. Run setup and script
echo "Running ML Pipeline on AWS EC2..."
ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ec2-user@${PUBLIC_IP} << 'INNER_EOF'
    set -e
    sudo dnf update -y
    sudo dnf install -y python3-pip python3-devel gcc-c++
    cd ~/ml-challenge
    pip3 install pandas numpy scipy scikit-learn lightgbm
    echo "Starting Python Script with maximum CPUs..."
    python3 hackathon_solution.py
INNER_EOF

# 6. Bring results back
echo "Downloading results..."
mkdir -p aws_output
rsync -Pav -e "ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no" ec2-user@${PUBLIC_IP}:~/ml-challenge/output/ ./aws_output/

# 7. Cleanup
echo "Cleaning up AWS resources to prevent extra charges..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION --profile $PROFILE > /dev/null
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID --region $REGION --profile $PROFILE
aws ec2 delete-security-group --group-id $SG_ID --region $REGION --profile $PROFILE
aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION --profile $PROFILE
rm -f ${KEY_NAME}.pem

echo "Done! Execution finished successfully. Results are in the aws_output/ directory."
