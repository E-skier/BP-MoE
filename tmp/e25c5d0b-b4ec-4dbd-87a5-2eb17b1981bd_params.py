datasets = [
    [
        dict(
            abbr='lukaemon_mmlu_college_biology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_biology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_college_chemistry',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_chemistry',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_college_computer_science',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_computer_science',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_college_mathematics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_mathematics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_college_physics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_physics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_electrical_engineering',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='electrical_engineering',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_astronomy',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='astronomy',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_anatomy',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='anatomy',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_abstract_algebra',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='abstract_algebra',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_machine_learning',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='machine_learning',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_clinical_knowledge',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='clinical_knowledge',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_global_facts',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='global_facts',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_management',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='management',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_nutrition',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='nutrition',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_marketing',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='marketing',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_professional_accounting',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='professional_accounting',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_geography',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_geography',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_international_law',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='international_law',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_moral_scenarios',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='moral_scenarios',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_computer_security',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='computer_security',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_microeconomics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_microeconomics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_professional_law',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='professional_law',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_medical_genetics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='medical_genetics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_professional_psychology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='professional_psychology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_jurisprudence',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='jurisprudence',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_world_religions',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='world_religions',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_philosophy',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='philosophy',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_virology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='virology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_chemistry',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_chemistry',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_public_relations',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='public_relations',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_macroeconomics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_macroeconomics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_human_sexuality',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='human_sexuality',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_elementary_mathematics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='elementary_mathematics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_physics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_physics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_computer_science',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_computer_science',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_european_history',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_european_history',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_business_ethics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='business_ethics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_moral_disputes',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='moral_disputes',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_statistics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_statistics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_miscellaneous',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='miscellaneous',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_formal_logic',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='formal_logic',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_government_and_politics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_government_and_politics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_prehistory',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='prehistory',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_security_studies',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='security_studies',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_biology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_biology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_logical_fallacies',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='logical_fallacies',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_world_history',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_world_history',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_professional_medicine',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='professional_medicine',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_mathematics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_mathematics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_college_medicine',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='college_medicine',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_us_history',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_us_history',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_sociology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='sociology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_econometrics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='econometrics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_high_school_psychology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='high_school_psychology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_human_aging',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='human_aging',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_us_foreign_policy',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='us_foreign_policy',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='lukaemon_mmlu_conceptual_physics',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='conceptual_physics',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
        dict(
            abbr='hellaswag',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        0:
                        dict(round=[
                            dict(prompt='{ctx}', role='HUMAN'),
                            dict(prompt='{A}', role='BOT'),
                        ]),
                        1:
                        dict(round=[
                            dict(prompt='{ctx}', role='HUMAN'),
                            dict(prompt='{B}', role='BOT'),
                        ]),
                        2:
                        dict(round=[
                            dict(prompt='{ctx}', role='HUMAN'),
                            dict(prompt='{C}', role='BOT'),
                        ]),
                        3:
                        dict(round=[
                            dict(prompt='{ctx}', role='HUMAN'),
                            dict(prompt='{D}', role='BOT'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='opencompass/hellaswag',
            reader_cfg=dict(
                input_columns=[
                    'ctx',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='label'),
            type='opencompass.datasets.HellaswagDataset'),
        dict(
            abbr='ARC-c',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='ARC-Challenge',
            path='opencompass/ai2_arc-dev',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.ARCDataset'),
        dict(
            abbr='ARC-e',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='ARC-Easy',
            path='opencompass/ai2_arc-easy-dev',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.ARCDataset'),
        dict(
            abbr='openbookqa',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt=
                                'Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='A', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt=
                                'Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='B', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt=
                                'Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='C', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt=
                                'Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='D', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='main',
            path='opencompass/openbookqa_test',
            reader_cfg=dict(
                input_columns=[
                    'question_stem',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.OBQADataset'),
        dict(
            abbr='openbookqa_fact',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt=
                                'Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='A', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt=
                                'Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='B', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt=
                                'Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='C', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt=
                                'Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:',
                                role='HUMAN'),
                            dict(prompt='D', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='additional',
            path='opencompass/openbookqa_fact',
            reader_cfg=dict(
                input_columns=[
                    'question_stem',
                    'A',
                    'B',
                    'C',
                    'D',
                    'fact1',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.OBQADataset'),
        dict(
            abbr='piqa',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        0:
                        'The following makes sense: \nQ: {goal}\nA: {sol1}\n',
                        1:
                        'The following makes sense: \nQ: {goal}\nA: {sol2}\n'
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='opencompass/piqa',
            reader_cfg=dict(
                input_columns=[
                    'goal',
                    'sol1',
                    'sol2',
                ],
                output_column='label',
                test_split='validation'),
            type='opencompass.datasets.PIQADataset'),
        dict(
            abbr='BoolQ',
            data_files='opencompass/boolq',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        0:
                        dict(round=[
                            dict(
                                prompt='{passage}\nQuestion: {question}?',
                                role='HUMAN'),
                            dict(prompt='No', role='BOT'),
                        ]),
                        1:
                        dict(round=[
                            dict(
                                prompt='{passage}\nQuestion: {question}?',
                                role='HUMAN'),
                            dict(prompt='Yes', role='BOT'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='json',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'passage',
                ],
                output_column='answer',
                test_split='train'),
            split='train',
            type='opencompass.datasets.BoolQDataset'),
    ],
]
models = [
    dict(
        abbr='entropy_only_seed778_200k_full_b8',
        batch_size=8,
        ckpt_dir=
        'runs/byte_entropy_repro_scale/byte_entropy_seed778_200000step/checkpoints/0000200000',
        consolidate_if_needed=True,
        entropy_attn_impl='sdpa',
        entropy_ckpt_dir=None,
        entropy_state_dict_path=None,
        generator_kwargs=dict(
            dtype='bf16',
            max_prompt_len=2048,
            max_tokens=2176,
            temperature=0.0),
        max_out_len=128,
        max_seq_len=2048,
        run_cfg=dict(num_gpus=1, num_procs=1),
        type='bytelatent.opencompass.ByteLatentOpenCompassModel'),
]
work_dir = '/home/ubuntu/BP-MoE/runs/opencompass/entropy_only_seed778_200k_full_b8/20260606_102155'
