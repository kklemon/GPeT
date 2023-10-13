from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        # parser.link_arguments(
        #     "data.tokenizer_config.lang",
        #     "model.tokenizer_config.lang",
        #     apply_on="instantiate"
        # )
        # parser.link_arguments(
        #     "data.tokenizer_config.dim",
        #     "model.tokenizer_config.dim",
        #     apply_on="instantiate"
        # )
        # parser.link_arguments(
        #     "data.tokenizer_config.vocab_size",
        #     "model.tokenizer_config.vocab_size",
        #     apply_on="instantiate"
        # )

        # parser.link_arguments(
        #     "data.tokenizer_config",
        #     "model.tokenizer_config",
        #     apply_on="instantiate"
        # )


        parser.link_arguments(
            "data.seq_len",
            "model.model_config.seq_len",
            apply_on="instantiate"
        )
