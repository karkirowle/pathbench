### COPAS Dataset Preparation

We processed the dataset to create standardized subsets for analysis, focusing primarily on two evaluation formats: **Word** and **Sentence** tasks. To systematically evaluate performance, the data was further partitioned into **balanced**, **unbalanced**, and **all-inclusive** configurations.

#### Word Task

The Word Task is derived from the *DIA* (Dutch Intelligibility Assessment) subset. Annotations were extracted from the provided TextGrids, which contain both intended and perceived transcriptions. To align with our experimental requirements, we utilized the intended transcriptions. All text was normalized to lowercase and stripped of punctuation to ensure consistency across the control and pathological groups.

Creating a balanced setting for this task is inherently challenging, as the original dataset was explicitly designed to be unbalanced.

* To construct the **balanced setting**, we restricted the data to speakers who completed the specific "A18" task, utilizing the intelligibility scores corresponding directly to that task.

* In contrast, for the **unbalanced** and **all** settings, we assigned each speaker their average intelligibility score calculated across all three available assessments.

#### Sentence Task

The Sentence Task consists of two fixed Dutch sentences:

* *S1*: "Wil je liever de thee of de borrel?"

* *S2*: "Na nieuwjaar was hij weeral hier."

Because there are no explicitly derived intelligibility scores for the continuous sentence recordings, we adopted the speakers' overall word-task scores as a proxy measure.