#!/bin/sh
# INFO page showing basic information

DIALOG_CANCEL=1
DIALOG_ESC=255
HEIGHT=0
WIDTH=0

function display_result() {
  dialog --title "$1" \
    --no-collapse \
    --msgbox "$result" 20 100
}

function fileChooser(){
  local __DIR=$1
  local __RESULT=$(dialog --clear --title "Choose Directory" --stdout \
                   --title "Choose File"\
                   --fselect $__DIR 14 58)

  echo $__RESULT
}

function selectModel(){
	#model_path=$(dialog --title --no-collapse "Create Directory" \
	#--inputbox "Enter the directory name:" 8 40)
	model_path=$(dialog --title "You know it" \
 --no-collapse \
    --inputbox "Choose your network Model:
    \n      (1) RSUNet(My Bravo Network) - UAV \n \
     (2) RSPNet - UAV \n \
     (3) FCN - UAV \n \
     (4) SegNet - UAV \n \
     Input the number please:
     " 20 100 --output-fd 1)
	# read f
	# if [ "$f" = "1" ]; then
	# 	model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
	# elif [ "$f" = "2" ]; then
	# 	model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
	# elif [ "$f" = "9" ]; then
	# 	model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
	# fi
}

dialog --backtitle "Liu Yu @ BUAA 2018 \n Pixel-wise Classification \
for Remote Sensing Imagery" --title "Info" --msgbox " This serves as the DEMO for \
LIU Yu's Master Thesis's Work in Beihang University 2018 \n\n Yu's Thesis is about Pixel-wise Classification on \
Remote Sensing Images using Caffe Framework. \n\n Press OK to continue...  " 15 75

# Return status of non-zero indicates cancel
if [ "$?" != "0" ]
then
	dialog --title "ESC" --msgbox "DEMO is stopeed" 10 50
else
	while true; do
	  exec 3>&1
	  selection=$(dialog \
	    --backtitle "Yu Liu @ Beihang University - Semantic Segmentation UAV -2018-" \
	    --title "Menu" \
	    --clear \
	    --cancel-label "Exit" \
	    --menu "Please select:" $HEIGHT $WIDTH 4 \
	    "1" "Select an Image to Split and Inference" \
	    "2" "Inference This Image with My Great CNN Model" \
	    "3" "Display the Inference Result" \
	    2>&1 1>&3)
	  exit_status=$?
	  exec 3>&-
	  case $exit_status in
	    $DIALOG_CANCEL)
	      clear
	      echo "Program terminated."
	      exit
	      ;;
	    $DIALOG_ESC)
	      clear
	      echo "Program aborted." >&2
	      exit 1
	      ;;
	  esac
	  case $selection in
	    0 )
	      clear
	      echo "Program terminated."
	      ;;
	    1 ) # select image to infer
	      result=$( fileChooser /media/lh/D/Data/Partion1/test/ )
	      img_select=$result
	      xdg-open $img_select # show the selected image
	      result=$( fileChooser /media/lh/D/Data/Partion1/ )
	      targ_folder=$result
	      display_result "Your Image Selection"
	      python /home/lh/Documents/HSNRS/script/data_preprocessing/split_one.py -i $img_select -t $targ_folder
	      read -p "Press [Enter] to go on..."
	      ;;
	    2 ) # select Model
	      selectModel
	      case $model_path in
	      	1)
			model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/RSnet_d128_infer.prototxt"
			weight_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
			;;
			2)
			model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/RSnet_d128_infer.prototxt"
			weight_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
			;;
			3)
			model_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/RSnet_d128_infer.prototxt"
			weight_path="/home/lh/Documents/HSNRS/net/UAV_hsnet/model/Unet_for_MSc.caffemodel"
			;;
	      esac
	      python /home/lh/Documents/HSNRS/script/network_training/UAV_infer_one.py -m $model_path \
	      -w $weight_path -i $img_select -t $targ_folder
	      read -p "Press [Enter] to go on..."
	      ;;
	    3 )
	      python /home/lh/Documents/HSNRS/script/network_training/UAV_assemble_one.py -t $targ_folder -i $img_select
	      python /home/lh/Documents/HSNRS/script/visualize/visualise_one.py -t $targ_folder -i $img_select
	      read -p "Press [Enter] to go on..."
	      result=$( fileChooser /media/lh/D/Data/Partion1/ )
	      xdg-open $result
	      read -p "Press [Enter] to go on..."
	      ;;
	  esac
	done
fi
