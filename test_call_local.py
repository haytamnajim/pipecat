import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")

system_instruction = """
Tu es un assistant vocal amical et serviable.
Tes réponses seront prononcées à voix haute, réponds de façon naturelle, concise et vivante en français.
Évite les puces, listes ou émoticônes.
"""


async def main():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Aucune clé GOOGLE_API_KEY ou GEMINI_API_KEY trouvée dans le .env !")
        return

    logger.info("Initialisation du transport audio local (Microphone & Haut-parleurs)...")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    llm = GeminiLiveLLMService(
        api_key=api_key,
        settings=GeminiLiveLLMService.Settings(
            system_instruction=system_instruction,
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.add_workers(worker)

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        logger.info(f"🎤 Utilisateur: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"🤖 Bot: {message.content}")

    logger.info("Démarrage de l'appel ! Vous pouvez parler directement dans votre micro.")
    context.add_message({"role": "developer", "content": "Dis bonjour et présente-toi brièvement."})
    await worker.queue_frames([LLMRunFrame()])

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
