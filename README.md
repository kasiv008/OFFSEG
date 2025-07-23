OFFSEG: A Semantic Segmentation Framework for Off-Road Driving

Official implementation of the paper:
"OFFSEG: A Semantic Segmentation Framework for Off-Road Driving"

    📄 Read the paper: arXiv Link

    🎥 Watch sample video on RELLIS-3D: Google Drive

🚧 OFFSEG Applied to NYCC Inspection Reports
<img width="1626" height="1176" alt="OFFSEG NYCC Demo" src="https://github.com/user-attachments/assets/872f014b-1460-464f-9c40-e1e5123ff134" />
📁 Project Structure & Training

    segmentation/: Contains implementations of BiSeNetV2 and HRNetV2+OCR used for training.

    Pipelines configured for the RUGD dataset.

    For re-training, follow the instructions from the respective upstream repositories.

🔗 Download pretrained weights:
Google Drive Link
▶️ Inference

Run the following script to generate segmentation masks:

python Pipeline/pipeline.py

    Note: Pretrained weights on the RUGD dataset yielded robust results on custom datasets as well.

🧠 Future Work

    Integrate instance segmentations with LLMs to generate automated inspection reports.

📚 Citation

If you find this work useful, please cite us:

@INPROCEEDINGS{9551643,
  author={Viswanath, Kasi and Singh, Kartikeya and Jiang, Peng and Sujit, P.B. and Saripalli, Srikanth},
  booktitle={2021 IEEE 17th International Conference on Automation Science and Engineering (CASE)}, 
  title={OFFSEG: A Semantic Segmentation Framework For Off-Road Driving}, 
  year={2021},
  pages={354-359},
  doi={10.1109/CASE49439.2021.9551643}
}

⚠️ Disclaimer

This repository is still under active development. Stay tuned for updates.
