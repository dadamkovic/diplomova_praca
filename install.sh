#!/bin/bash
user_inp=""
while [[ "$user_inp" != "yes" && "$user_inp" != "no" ]]
do
	read -p "Do you want to install Lambda Stack? (yes/no): " user_inp
done

if [ "$user_inp" = "yes" ]
then
	LAMBDA_REPO=$(mktemp) && \
	wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
	sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
	sudo apt-get update && sudo apt-get install -y lambda-stack-cuda
fi
pip3 install wheel
pip3 install -r requirements.txt
pip3 install -e ./code/nextro_env
echo "------------------------"
echo "Install complete!"
if [ "$user_inp" = "yes" ]
then
	read -p "Restart the computer now to complete the installation of lambda stack? (yes/no): " user_inp

	while [[ "$user_inp" != "yes" && "$user_inp" != "no" ]]
	do
		read -p  "Restart the computer now to complete the installation of lambda stack? (yes/no): " user_inp
	done

	if [ "$user_inp" = "yes" ]
	then
		sudo reboot
	fi
fi
