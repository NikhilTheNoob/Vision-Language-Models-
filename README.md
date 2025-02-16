# Vision-Language-Models
For the 2024 WiDS project on Vision Language Models which involves implementing the papers CLIP (Contrastive Learning Image Pre-Training), CoOp (Context Optimization), and Imbalanced VLM

Paper 1 - CLIP (Contrastive Learning Image Pre-Training)

• Central idea: Learn visual representations from natural language supervision by aligning images with their textual descriptions rather than relying on fixed, manually labeled classes
• Data: Uses a massive dataset of 400 million image-text pairs sourced from the internet, offering a broader and more diverse supervision signal compared to traditional datasets like ImageNet
• Approach – Contrastive Learning:
• Jointly trains an image encoder and a text encoder
• Uses a contrastive objective (InfoNCE loss) to pull together correct (image, text) pairs while pushing apart mismatched pairs within each batch
• Zero-shot classification:
• Synthesizes classification weights dynamically by feeding textual descriptions (e.g., “A photo of a [CLASS].”) into the text encoder
• Enables the model to generalize to new tasks without additional training
• Model scaling:
• Explores different architectures, including modified ResNets (e.g., ResNet-50x4, ResNet-50x16, ResNet-50x64) and Vision Transformers
• Demonstrates that performance scales predictably with increased compute and model capacity (measured in GFLOPs)
• Evaluation insights:
• Uses linear probe evaluations (fitting a simple linear classifier on fixed features) to measure representation quality
• Compares zero-shot performance to fully supervised and few-shot methods, often matching or exceeding them on benchmarks like ImageNet
• Prompt engineering:
• Highlights the importance of natural language prompts in synthesizing classifiers, where even small tweaks (e.g., “A photo of a [CLASS].”) can have significant impacts on performance
• Robustness and generalization:
• The model shows strong robustness to distribution shifts and adapts well to a variety of tasks including fine-grained classification, OCR, geo-localization, and action recognition
• The use of natural language allows the model to handle polysemy and capture a wider array of visual concepts compared to rigid, fixed labels
• The contrastive loss formulation simplifies training by avoiding the need to generate detailed text captions, focusing instead on matching global image-text pairs
• The scalability of the approach is demonstrated through systematic experiments with different architectures and compute budgets, underscoring the link between model capacity and transfer performance
• Overall, the CLIP approach illustrates how leveraging vast, naturally occurring image-text pairs can yield a versatile, zero-shot capable visual learner that competes with traditional supervised methods on many tasks
Paper 2 - Context Optimization (CoOp)
 • Central idea: Adapt large pre-trained vision-language models (like CLIP) for downstream image recognition by automating prompt engineering rather than manually tuning text prompts
• Motivation: Manual prompt tuning is time-consuming and requires domain expertise; even slight changes in wording can have a significant impact on performance
• Approach – Context Optimization (CoOp):
	• Models the prompt’s context words with learnable continuous vectors while keeping       	  the massive pre-trained parameters fixed
• Provides two designs:
	• Unified context: A single set of context vectors shared across all classes
	• Class-specific context (CSC): Unique context vectors per class, useful for some  	  fine-grained tasks
• Implementation details:
	• The prompt is structured as a combination of learned context tokens followed by the  	  class token (e.g., [V]₁, [V]₂, …, [V]ₘ [CLASS]), with flexibility to position the class  	  token at the end or in the middle
	• Trained via standard cross-entropy loss, where gradients are back-propagated   	  through the frozen text encoder to optimize these context vectors
• Key experimental insights:
• CoOp is highly data-efficient, beating hand-crafted prompts with as few as 1–2 training shots
• With more shots (e.g., 16), average performance gains can be around 15%, and in some specialized tasks (like EuroSAT), improvements exceed 45%
• Outperforms a linear probe baseline, especially in very low-data regimes
• The method reveals that learnable prompts can capture task-relevant context automatically, reducing the need for expert manual tuning
• Analysis of context length shows a trade-off: more context tokens generally improve performance but might hurt robustness to distribution shifts if overfitting occurs
• When comparing unified versus class-specific context, unified context tends to work better for generic object and scene recognition, while CSC can offer advantages for fine-grained categories
• CoOp’s robustness to domain shifts is notable—it outperforms both zero-shot (hand-crafted prompts) and linear probe methods on diverse target distributions
• Initialization of the context vectors (random vs. using embeddings from “a photo of a”) doesn’t make a significant difference, suggesting the optimization process is robust
• Even though the learned context vectors are continuous and hard to interpret directly, an indirect analysis (searching for nearest vocabulary words) hints that they capture meaningful, task-relevant information
• Overall, CoOp demonstrates a simple yet powerful way to adapt vision-language models for varied downstream tasks with minimal data, while also offering strong robustness to distribution changes
Paper 3 - Imbalance VLM 
• Central idea: Explore how vision-language models (VLMs) like CLIP can be adapted for imbalanced learning, addressing their weak performance on datasets where class distributions are skewed (e.g., tail classes have very few examples)
• Motivation:
• Zero-shot VLMs perform well on balanced benchmarks but struggle on imbalanced datasets (e.g., CLIP achieves only 5% on iNaturalist18)
• Real-world applications (e.g., medical diagnosis, autonomous driving) demand high accuracy on rare classes
• Proposed modifications:
• Introduce a lightweight decoder after the frozen image encoder to capture subtle features of tail classes while avoiding out-of-memory issues caused by many classes
• Integrate various imbalanced learning strategies such as prompt tuning, fine-tuning, and specialized loss functions (e.g., Focal Loss, Balanced Softmax, LDAM Loss)
• Approach details – Methodology:
• Zero-shot classification: Uses text queries (e.g., “a picture of a [Class]”) to compute cosine similarity between image and text embeddings
• Fine-tuning: Replace CLIP’s visual projection layer with a simple classifier; however, this often underperforms because it fails to capture nuances for minority classes
• Incorporation of imbalanced methods:
• Add the lightweight decoder to extract better representations
• Employ loss function engineering (e.g., CBW, Focal Loss) and two-stage training methods (e.g., CRT, MARC) to balance gradients across head and tail classes
• Key experimental insights:
• Significant improvements when using decoder plus imbalanced algorithms: For instance, on iNaturalist18, accuracy boosts from 5.45% (zero-shot) to nearly 70% with proper fine-tuning
• On ImageNet-LT and Places-LT, performance gains of 6–7% over zero-shot baselines are reported
• The method outperforms both linear probing and full fine-tuning in capturing nuanced features for minority classes
• The decoder is lightweight (built with a few transformer blocks) and adds only a moderate increase in GPU memory usage, making it feasible on consumer GPUs
• Experiments reveal that simply increasing pre-training data does not necessarily improve performance on imbalanced tasks—the tuning and adaptation strategy is critical
• The study emphasizes the trade-off between representation quality and classifier adjustment in long-tailed scenarios, highlighting the importance of using specialized loss functions to rebalance gradients
• Analyses include comparisons across various backbones (e.g., ViT-B16 vs. ViT-L14), pre-training datasets (CLIP vs. Laion-CLIP), and training cost metrics (GPU memory, power consumption, carbon footprint)
• The paper outlines a comprehensive training procedure that combines instance-balanced sampling in the first stage with classifier re-calibration (via additional trainable parameters) in the second stage
