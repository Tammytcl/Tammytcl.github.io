---
layout: archive
title: "Curriculum Vitae"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
- **Undergraduate**, 2022 - Present 
  - Shandong University
  - **Major**: Computer Science and Technology
  - **GPA**: 90.132/100     
  - **Rank:** 4/106
  - **English Proficiency**: CET4 607, CET6 575


---

# Research Experience

## 心脏图像合成与分割研究 Research on Cardiac Image Synthesis and Segmentation
### August 2024 - October 2024
- **目标 (Objective):** Proposed a Feature Quantization-based Cardiac Diffusion Model (FQCDM) to address issues of data scarcity, noise, and class imbalance in cardiac image datasets, ultimately enhancing the performance and robustness of cardiac image segmentation. 提出一种基于特征量化的心脏图像合成模型，以应对心脏图像数据集中的样本稀缺、噪声和类别不平衡问题，提升心脏图像分割的性能和鲁棒性。
- **创新(Innovation):**
  1.	Developed the Feature Quantization-based Cardiac Diffusion Model (FQCDM), integrating Denoising Diffusion Probabilistic Model (DDPM) with feature quantization to generate high-quality cardiac images. 提出特征量化心脏扩散模型（FQCDM），结合去噪扩散概率模型（DDPM）和特征量化生成高质量的心脏图像。
  2.	Designed a Dual-Branch Discriminator (DBD) for simultaneous evaluation of global features and edge information, enhancing the diversity and quality of generated images. 设计双分支GAN判别器（DBD），同时评估生成图像的全局特征和边缘信息，提高生成图像的多样性和质量。
  3.	Proposed three strategies to improve segmentation performance using synthetic cardiac images: mixed training, self-supervised pre-training, and integration with data augmentation for optimized cardiac segmentation models. 提出三种利用合成心脏图像提升分割性能的策略：混合训练、自监督预训练以及数据增强结合，以优化现有心脏图像分割模型。
- **效果(Achievement):**  Experiments demonstrate that FQCDM can generate high-quality cardiac images with similar distribution properties, significantly enhancing segmentation performance on the MMWHS-2017 and ACDC cardiac datasets. The inclusion of synthetic data and the application of the three training strategies yielded substantial accuracy improvements, confirming the feasibility and effectiveness of synthetic cardiac data in cardiac segmentation tasks. 实验表明FQCDM能够生成高质量且分布相似的心脏图像，显著提升了在MMWHS-2017和ACDC等心脏数据集上的分割性能。特别是通过合成数据的引入和三种训练策略的应用，实现了显著的分割精度提升，验证了合成心脏数据在心脏图像分割任务中的可行性和有效性。
- **发表状态(Publication Status):**  Manuscript under review in the IEEE Journal of Biomedical and Health Informatics (JBHI, Category 1, Chinese Academy of Sciences) titled “FQCDM: Feature Quantization-Based Cardiac Image Diffusion Synthesis Model.” 

## 医学影像分割 Medical Image Segmentation
### August 2024 - October 2024
- **目标 (Objective):** Developed a parallel CNN and Mamba architecture to balance parameter efficiency, computational complexity, and segmentation accuracy. 开发了一种并行的CNN和Mamba架构, 实现了平衡参数量、运算复杂度和分割效果。
- **创新(Innovation):**
	1.	Introduced the PCMamba architecture, combining Mamba’s long-range modeling capabilities with the convolutional advantage for capturing fine-grained structures, significantly improving segmentation performance at both coarse and fine granularities. 我们探索性地结合了Mamba在长距离建模方面的卓越能力与卷积在细微结构捕捉中的优势，提出了PCMamba架构，通过并行处理显著提升了医学图像分割的粗细粒度性能。
	2.	Designed two innovative modules, the DiscWideFusion module and RepLKWideFusion module, to optimize feature extraction: DiscWideFusion uses dilated convolutions to capture multi-scale features while minimizing parameters and computational costs; RepLKWideFusion combines small and large kernels for enhanced precision in capturing subtle structures. 我们引入了DiscWideFusion模块和RepLKWideFusion模块，分别利用离散卷积和多尺寸卷积组合来优化多尺度特征提取，提高了模型精度并降低了参数量。
	3.	Developed a gated adaptive loss function that dynamically adjusts training parameters based on model performance, leading to significant improvements in segmentation accuracy for challenging regions, such as low-contrast boundaries. 设计了门控自适应损失函数，根据模型表现动态调整训练参数，有效提升复杂区域的分割精度。
- **效果(Achievement):**  Achieved state-of-the-art results across multiple single and multi-organ segmentation tasks on ACDC, MMWHS-CT, LAHeart2018, TN3K, DDTI, Synapse, ISIC2016, ISIC2017, and PH2 datasets, demonstrating the model’s superior performance across diverse organs and imaging modalities. 在ACDC、MMWHS-CT、LAHeart2018、TN3K、DDTI、Synapse、ISIC2016、ISIC2017、PH2多个器官的单/多语义分割任务上取得了SOTA效果, 证明了我们的模型在不同器官不同成像模式下的效果优越。
- **发表状态(Publication Status):**  Manuscript under review in IEEE Transactions on Medical Imaging (TMI, Category 1, Chinese Academy of Sciences) titled “PCMamba: Parallel Convolution-Mamba Network for Medical Image Segmentation.” 一区期刊IEEE Transactions on Medical Imaging（TMI）中科院一区在投，论文名字为“PCMamba: Parallel Convolution-Mamba Network for Medical Image Segmentation”。

## 心脏医学影像分割 Cardiac Medical Image Segmentation
### March 2024 - August 2024
- **目标 (Objective):** Developed a novel cardiac segmentation model, PSVT, integrating the Transformer architecture with CNNs for efficient feature extraction. 开发了一种名为PSVT的新型心脏分割模型，将Transformer架构与CNNs融合，以提高特征提取效率。
- **创新(Innovation):**
	1.	Introduced a biased window generation method and depthwise convolution to optimize both global and local feature recognition, enhancing computational efficiency and reducing memory consumption. 提出了独特的偏置窗口生成方法和深度卷积，优化了全局和局部特征识别，提高了计算效率并减少了内存占用。
	2.	Refined the attention mechanism by combining the Swin-Transformer-v2 attention module with convolutional blocks and integrating DW-conv layers into the feedforward layer. This architecture boosts the model’s feature extraction capabilities for both local and global patches. 在注意力机制中结合Swin-Transformer-v2的注意力模块与卷积模块，并在前馈层中集成了DW-conv层，增强了对局部和全局特征的提取能力。
	3.	Redesigned the Patch Merging and Patch Expanding modules using transposed convolutions, improving the integration of multi-scale features in the upsampling and downsampling stages. 重构了Patch Merging和Patch Expanding模块，采用转置卷积提高了多尺度特征的集成效果。
- **效果(Achievement):**  Achieved superior results on the ACDC, MMWHS-CT, and LASEG-2013 datasets, outperforming contemporary segmentation methodologies. 在ACDC、MMWHS-CT和LASEG-2013数据集上取得了优异的结果，超越了当代的分割技术。
- **发表状态(Publication Status):**  Manuscript under review in Biomedical Signal Processing and Control (Category 2, Chinese Academy of Sciences) with the title “PSVT: Pyramid Shifted Window based Vision Transformer for Cardiac Image Segmentation.”

## 心脏医学影像分割 Cardiac Medical Image Segmentation  
### December 2023 - May 2024
- **目标 (Objective):** Designed an advanced cardiac segmentation network employing the Multi-Scale, Multi-Head Self-Attention (MSMHSA) mechanism for enhanced feature extraction. 设计了一个采用多尺度多头自注意力（MSMHSA）机制的先进心脏分割网络，增强特征提取能力，以提高心脏结构的分割精度。
- **方法 (Approach):** Integrated MSMHSA into the DeepLab V3+ architecture, leveraging its strengths in capturing contextual information and improving segmentation accuracy through decoder skip connections. 将MSMHSA集成到DeepLab V3+架构中，利用其在捕获上下文信息和通过解码器跳跃连接提高分割精度方面的优势。通过在不同尺度下引入三个独立的MHSA机制，进一步增强了对心脏亚结构的精确分割能力。
- **成果 (Outcome):**   Published in “J. Imaging 2024, 10, 135” with the title “MSMHSA-DeepLab V3+: An Effective Multi-Scale, Multi-Head Self-Attention Network for Dual-Modality Cardiac Medical Image Segmentation.” 


## 三维空间自由弯管一体化成型研究 Research on Integrated Forming of 3D Spatial Free Bending Pipe
### October 2023 - April 2024
- **目标 (Objective):** Investigated the integrated forming process of 3D spatial free bending pipes, utilizing a springback prediction model for compensation to enhance forming accuracy. 研究了三维空间自由弯管的一体化成型技术，开发了一种基于反弹预测模型的补偿策略，以提高成型精度。
- **主要工作 (Key Activities):** Engaged in computer simulation, algorithm optimization, and result visualization to ensure the precision and efficiency of the forming process. 进行了计算机模拟仿真、算法优化和结果可视化，确保了成型过程的精确性和效率。
- **创新 (Innovation):** Developed a springback compensation strategy based on predictive modeling, contributing to the improvement of forming quality and reduction of post-forming adjustments. 制定了一项反弹补偿策略，基于预测模型，有助于改善成型质量和减少成型后的调整工作。
- **成果 (Outcome):** 产出为一项软件著作权，名称为《渐进式三维自由弯曲成形操作系统》(Progressive 3D Free Bending Forming Operating System)，已注册登记。


---


Awards
======
* 2024 National College Students' Software Innovation Competition (2024 年全国大学生软件创新大赛) - First Prize (Second in the Nation) 全国一等奖第二名
    * Designed MelodyRNN to infer users' rhythmic and musical preferences from their input, dynamically adjusting and learning the model accordingly. 设计 MelodyRNN 根据用户的输入推断用户对应节奏和音乐的喜好，进行动态的模型调整和学习
    * Introduced the MusicTransformer model for secondary processing on the melodies generated by MelodyRNN, implementing a more delicate and in-depth learning of musical features through an encoder-decoder structure, resulting in melodies more suitable for children's basic musical foundations. 引入了 MusicTransformer 模型在 MelodyRNN 生成的旋律基础上进行二次处理,通过编码器-解码器结
      构实现了对音乐特征的更加细腻和深入的学习。得到更符合儿童的基础音乐旋律。
    * Utilized MusicGen to increase the controllability of the generated samples by introducing unsupervised melody adjustment, allowing users to more flexibly control the melodic part of the generated music, thus achieving personalized customization and adjustment of the generated music. 利用 MusicGen 通过引入无监督的旋律调节，增加提高了生成样本的可控性。用户可以更加灵活地控制
      生成音乐的旋律部分，从而实现对生成音乐的个性化定制和调整。
* The 17th China College Students Computer Design Competition /第17届中国大学生计算机设计大赛 - National Second Prize 全国二等奖
* 2024 China-US Youth Maker Competition / 2024 年中美青年创客大赛 - National Third Prize 全国三等奖
* 2024 China College Students Mathematical Competition / 2024 年全国大学生数学竞赛 - National Second Prize 全国二等奖
* 2023 National College Students Mathematical Modeling Competition / 2023 年全国大学生数学建模竞赛 - Provincial First Prize 省级一等奖
* 2024 Mathematical Contest in Modeling (MCM) / Interdisciplinary Contest in Modeling (ICM) /
  2024年美国大学生数学建模竞赛（MCM/ICM） - Honorable Mention H奖
* The 15th National College Student Electrical Mathematics Modeling Competition / 第十五届全国大学生电工数学建模竞赛 - Second Prize 二等奖
* 2022 Mathorcup University Mathematical Modeling Challenge Competition Big Data Competition Undergraduate Group / 2022 年 Mathorcup 高校数学建模挑战赛大数据竞赛本科生组 - Second Prize 二等奖
* The 4th "HuaShu Cup" National College Student Mathematical Modeling Competition Undergraduate Group / 第四届“华数杯”全国大学生数学建模竞赛 本科生组 - Second Prize 二等奖
* Future Designer National College Digital Art & Design Competition / 未来设计师·全国高校数字艺术设计大赛 - National Second Prize 全国二等奖
* 2024 China College Computer Competition - AIGC Innovation Competition / 2024中国高校计算机大赛·AIGC创新赛 - National Second Prize 全国二等奖





Scholarships and Honorary Titles
======
* Shandong University Special Talent Award (Entrepreneurship Practice, Research Innovation) 山东大学 特长奖(创业实践类、研究创新类)
* Shandong University First-Class Excellent Student Scholarship 山东大学 优秀学生一等奖
* Shandong University University-Level Merit Student 山东大学 校级三好学生
* Shandong University May Fourth Advanced Individual 山东大学 五四先进个人
* Shandong University 2023 Annual Innovation and Entrepreneurship Activities Advanced Individual 山东大学2023年度创新创业活动先进个人
* Outstanding Individual in Social Practice at Shandong University 山东大学 校级社会实践优秀个人

---


# Software Copyright / Patent
* **Patent Application**: One pending application for an invention patent, entitled "A Comprehensive Children's Music Education System." 发明专利 1 份在申，专利名称《一种综合性儿童音乐教学系统》
* **Software Copyright 1**: One registered copyright for a software work, entitled "The Tale Of Sound Art - A Children's Music Enlightenment Assistant Software Based on MR Glasses Interaction." 软件著作名称《音艺物语-基于 MR 眼镜交互的儿童音乐启蒙辅助软件》
* **Software Copyright 2**: Registered for a software work, entitled "Progressive 3D Free Bending Forming Operating System." 软件著作名称《渐进式三维自由弯曲成形操作系统》


---

# Internship Experience
## Summer 2023: Questionnaire Analysis Research Assistant
- **Institution**: Shandong University (山东大学) in collaboration with Shenzhen General Science Technology Education Development Research Center (深圳市通识科技教育发展研究中心)
- **Position**: Research Assistant specializing in questionnaire analysis
- **Key Responsibilities**:
  - Conducted a systematic analysis of questionnaires to assess the innovative capabilities and growth environments of children in remote rural areas, providing valuable insights into educational and developmental needs. (负责对偏远乡村儿童的创新能力和成长环境进行问卷调查分析，为了解教育和发展需求提供了宝贵见解。)
  - Collaborated with multidisciplinary teams to interpret data, ensuring the research's methodological rigor and relevance to the target population. (与多学科团队合作解读数据，确保研究的方法严谨且与目标人群相关。)
- **Supervisor**: Professor Gu Wu (吴贾教授)

## November 2023 - May 2024: Undergraduate Research Assistant (URAP)
- **Project Title**: Medical Volumetric Based on Machine Learning (基于机器学习的医学体绘制)
- **Institution**: Shandong University (山东大学)
- **Key Responsibilities**:
  - Engaged in a pioneering project applying machine learning algorithms to medical volumetry, aiming to enhance diagnostic accuracy and efficiency in clinical settings. (参与应用机器学习算法于医学体积测量的开创性项目，旨在提高临床环境中的诊断准确性和效率。)
  - Assisted in the development and refinement of machine learning models, including data preparation, feature extraction, and model training, under the guidance of a seasoned academic mentor. (在经验丰富的学术导师的指导下，协助开发和完善机器学习模型，包括数据准备、特征提取和模型训练。)
- **Mentor**: Associate Professor Fei Yang(杨飞副教授)

## October 2024 - November 2024: Undergraduate Research Assistant (URAP)
- **Project Title**:  Development of AI Recognition System for Raw Materials (荒料AI识别系统的研发)
- **Institution**: Shandong University (山东大学)
- **Key Responsibilities**:
  - Implemented inverse perspective transformations to process images, enabling the generation of top-down views for improved analysis of raw materials.
  - 利用逆透视变换处理图片，获取俯视图以改善对原材料的分析。
  - Employed the YOLOv5 model to train AI algorithms for the recognition of raw materials, facilitating the accurate determination of their volumes.
  - 利用YOLOv5模型进行AI算法训练，以识别原材料并准确确定其体积。
  - Conducted data preprocessing and augmentation to enhance the training dataset, improving the model’s performance and robustness.
  - 进行数据预处理和增强，以提升训练数据集，提高模型的性能和鲁棒性。
- **Mentor**: Associate Professor Yong Liu (刘勇副教授)

---

# Skills
* **Reading Enthusiast** - Passionate about reading a variety of books across literature, history, technology, and more, continuously enhancing my knowledge and cultural literacy. 阅读爱好者 - 热衷于阅读各类书籍，涵盖文学、历史、科技等多个领域，不断提升自我知识水平和文化素养。
* **Sports Lover** - Enjoy playing basketball and table tennis. I appreciate the joy of teamwork and competition. 运动爱好者 - 享受团队合作与竞技的快感。
* **Musical Theatre Enthusiast** - Deeply interested in musical theatre, appreciating the artistic charm of the combination of music and drama.音乐剧爱好者 - 对音乐剧有深厚兴趣，欣赏音乐剧的艺术魅力，享受音乐与戏剧的完美结合。
* **Guitar Learner** - Currently learning to play the guitar. My progress may not yet be concert-ready, but I appreciate the journey! 吉他在学，目前正在学习吉他，虽然水平还未达到演出标准，但我很享受这个过程！




---
# Other Awards(一些很水的奖)
- The 18th iCAN Innovation Contest (Shandong) (2024), Provincial Third Prize. (第十八届iCAN大学生创新创业大赛山东赛区三等奖 2024年)
- The 9th China International College Students' Innovation Contest (2024), National Gold Award. (第九届中国国际大学生创新大赛/互联网+ 全国赛金奖 2024年)
- The 9th China International College Students' Innovation Contest (2024), Shandong Province Gold Award. (第九届中国国际大学生创新大赛/互联网+ 山东省金奖 2024年)
- "Goertek Cup" 19th Shandong University Intelligent Innovation and Smart Creative Design Competition for Mechanical and Electrical Products (2024), First and Third Prizes. ("歌尔杯"第十九届山东大学机电产品智能创新与智慧创意设计竞赛 一、三等奖 2024年)
- The 15th Shandong University College Students' Energy Conservation and Emission Reduction Social Practice and Science and Technology Competition (2024), Second and Third Prizes. (山东大学第十五届大学生节能减排社会实践与科技竞赛 二、三等奖 2024年)
- 2024 Shandong University 4th "Smart Future (Himile Cup)" Innovation Competition,Innovation Track Award. (2024年山东大学第四届智造未来豪迈杯创新大赛 创新赛道获奖)
- The 3rd National College Students' Career Development Competition (2023), First Prize. (2023年第三届全国大学生职业发展大赛 一等奖)
- The 4th College Students' Organizational Management Skills Competition (2023), Second Prize. (2023年第四届大学生组织管理能力大赛 二等奖)
- Shandong University "Youth Dream Building, Progressing with the Times" Talent Development Lecture Series (Personal Growth Experience Category Outstanding Lecture Team). (山东大学“青春筑梦，与时偕行”成才大讲堂 个人成长经历类优秀主讲团队)
- "UN Procurement Cup" National College Students' English Vocabulary Competition (2022), Undergraduate Group First Prize. (2022年“联合国采购杯”全国大学生英语词汇大赛 本科组一等奖)
- "IEERA Cup" International Collegiate English Translation Challenge (2023), Undergraduate Group National Area Second Prize. (2023年“IEERA杯”国际高校英语翻译挑战赛 本科生组国区二等奖)
- "Innovation Practice Cup" National College Students' English Vocabulary Competition (2022), Third Prize. (2022年“创新实践杯”全国大学生英语词汇竞赛 三等奖)
- Shandong University "To Those Teachers Who Have Warmed Me" Three-Line Poetry Competition (2024) (山东大学“致那些温暖过我的老师”三行诗评选活动, 2024):
  - Received Second Prize for the poem "The Fragrance of Ink Endures" (《墨香长存》).
  - Received Third Prize for the poem "The Spring Blossom of My Teacher's Grace" (《恩师春华》).
- Shandong University "To the Lovely Me and You" Three-Line Poetry Competition (山东大学“致可爱的我和你”三行诗评选活动) Third Prize 
- Shandong University Winter Vacation Reading Challenge (2023) (山东大学寒假读书不孤单阅读打卡活动, 2023) Second Prize
- Shandong University Winter Vacation Plank Challenge (2024) (山东大学寒假平板支撑挑战赛, 2024) Third Prize in the Men's Division. (男子组获得三等奖)
- Shandong University Winter Vacation Fitness Challenge (2024) (山东大学寒假健身挑战赛, 2024)  Third Prize in the Men's Division. (男子组获得三等奖)

---

# Social Practice and Service
## Social Practice Awards
- "Qu Jing Tong You — Intelligent Progressive 3D Spatial Bending Machine" - Received the University-Level Outstanding Social Practice Award. 曲径通优——智能化渐进式三维空间弯管机，获得校级优秀社会实践奖
- "Chinese Herbal Medicine Cultivation in the Countryside, Reflecting People's Livelihood, Forging Rural Soul" - Received the University-Level Outstanding Social Practice Award and the Excellent Team Award for the "Three into the Country" Activity. 中药耕乡情，映民生，铸乡魂，获得校级优秀社会实践奖、三下乡优秀团队奖

## 2023 Shandong University Labor Service Practice Position
- Engaged in "Reader Information Platform Planning and Common Problem Organization" for the 2023 Shandong University Labor Service Practice Position. 参与2023年度山东大学劳动服务实践岗，负责读者资讯平台规划以及常见问题整理

## Social Service Highlights
- **Volunteer Service**: Accumulated over a hundred hours of volunteer service, actively participating in community engagement and various charitable activities. 拥有上百小时的志愿服务时长，积极参与社区服务和各类公益活动
- **Educational Outreach**: Actively involved in primary school campus visitation and demonstration events, contributing to educational outreach and community involvement. 积极参与小学生校园参观展示活动，致力于教育推广和社区参与
- **Technical Support**: Member of the Geek Bird Computer Repair Club, applying technical expertise to support and assist others. 极客鸟电脑维修社团成员，运用技术专长为他人提供支持与帮助
- **Academic Assistance**: Joined the Beidou (Big Dipper) Volunteer Team, actively providing tutoring services to classmates and sharing study notes. 加入北斗星志愿者团队，积极为同学提供答疑服务，分享学习笔记
- **Mental Health Certification**: Certified mental health committee member and qualified psychological therapist by the University Student Mental Health Committee Work Platform. (高校心理委员工作平台认证的心理委员和心理疗法的合格证书