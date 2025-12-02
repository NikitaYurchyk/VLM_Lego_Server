"""
Workflow Logic Module

This module contains all the workflow nodes and routing logic for the LEGO assembly process.
It handles message classification, step navigation, image processing, and feedback generation.
"""

import asyncio
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from .models import AssemblyState


async def call_llm_with_timeout(llm, messages, timeout_seconds=60):
    """Call LLM with timeout and return appropriate error message if timeout occurs"""
    try:
        return await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"The AI provider took too much time to respond (>{timeout_seconds} seconds). Please try again or check your connection."
        )


class WorkflowManager:
    """Manages the LangGraph workflow for LEGO assembly"""

    def __init__(self, agent):
        self.agent = agent  # Reference to main agent for access to LLMs and services
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for LEGO assembly"""
        workflow = StateGraph(AssemblyState)

        workflow.add_node("retrieve_instructions", self.retrieve_instructions)
        workflow.add_node("classify_message", self.classify_message)
        workflow.add_node(
            "classify_regular_instruction", self.classify_regular_instruction
        )
        workflow.add_node("process_image", self.process_image)
        workflow.add_node("compare_assembly", self.compare_assembly)
        workflow.add_node("generate_feedback", self.generate_feedback)
        workflow.add_node("update_progress", self.update_progress)
        workflow.add_node("complete_step", self.complete_step)
        workflow.add_node("next_step", self.next_step)
        workflow.add_node("prev_step", self.prev_step)
        workflow.add_node("reset_assembly", self.reset_assembly)
        workflow.add_node("zoom_in", self.zoom_in)
        workflow.add_node("zoom_out", self.zoom_out)
        workflow.add_node("rotate_left", self.rotate_left)
        workflow.add_node("rotate_right", self.rotate_right)
        workflow.add_node("show_image", self.show_image)
        workflow.add_node("repeat_instruction", self.repeat_instruction)
        workflow.add_node("current_step", self.get_current_step)

        workflow.set_entry_point("classify_message")

        workflow.add_conditional_edges(
            "classify_message",
            self.route_by_message_type,
            {
                "compare_assembly": "process_image",
                "classify_regular_instruction": "classify_regular_instruction",
            },
        )

        workflow.add_edge("process_image", "retrieve_instructions")
        workflow.add_edge("retrieve_instructions", "compare_assembly")
        workflow.add_conditional_edges(
            "compare_assembly",
            self.should_continue_or_provide_feedback,
            {
                "feedback_needed": "generate_feedback",
                "step_complete": "complete_step",
                "continue": "update_progress",
            },
        )
        workflow.add_conditional_edges(
            "classify_regular_instruction",
            self.route_by_regular_instruction,
            {
                "next_step": "next_step",
                "prev_step": "prev_step",
                "reset_assembly": "reset_assembly",
                "zoom_in": "zoom_in",
                "zoom_out": "zoom_out",
                "rotate_left": "rotate_left",
                "rotate_right": "rotate_right",
                "show_image": "show_image",
                "repeat_instruction": "repeat_instruction",
                "current_step": "current_step",
            },
        )

        workflow.add_edge("next_step", END)
        workflow.add_edge("prev_step", END)
        workflow.add_edge("reset_assembly", END)
        workflow.add_edge("zoom_in", END)
        workflow.add_edge("zoom_out", END)
        workflow.add_edge("rotate_left", END)
        workflow.add_edge("rotate_right", END)
        workflow.add_edge("show_image", END)
        workflow.add_edge("repeat_instruction", END)
        workflow.add_edge("current_step", END)
        workflow.add_edge("generate_feedback", END)
        workflow.add_edge("complete_step", END)
        workflow.add_edge("update_progress", END)
        return workflow.compile()

    # Message Classification Nodes
    async def classify_message(self, state: AssemblyState):
        """Classify incoming user messages"""
        print("classify_message started")

        user_message = state.user_message or "check"
        print(f"🔍 Classifying message: '{user_message}'")

        prompt = f"""Classify the message. If the message contains next, next step, previous, prev, reset, zoom in, zoom out or show image, they should be classified it as 'regular'. If the message contains word check, it should be classified as 'compare'.

Message: "{user_message}"

Respond with ONLY one word: either "check" or "regular"""

        try:
            result = await call_llm_with_timeout(
                self.agent.classify_msg_llm, [HumanMessage(content=prompt)]
            )

            # Parse simple text response
            response_text = result.content.strip().lower()
            message_type = "compare" if "compare" in response_text else "regular"

            print(f"🔍 Classification result: {message_type}")
            state.message_type = message_type
        except TimeoutError as e:
            print(f"Timeout classifying message: {e}")
            # Default fallback classification
            state.message_type = (
                "compare" if "compare" in user_message.lower() else "regular"
            )
            raise  # Re-raise to be handled by the parent handler
        except Exception as e:
            print(f"Error classifying message: {e}")
            # Default fallback classification
            state.message_type = (
                "compare" if "compare" in user_message.lower() else "regular"
            )

        return state

    async def classify_regular_instruction(self, state: AssemblyState):
        """Classify regular instruction messages"""
        print("classify_regular_instruction started")

        user_message = state.user_message or "next"

        prompt = f"""Classify the message into one of these categories:
- If the message contains words "next" or "next step" classify it as "next"
- If the message contains words "previous" or "prev", classify it as "prev"
- If the message contains words "reset" or "start over", classify it as "reset"
- If the message contains words "zoom in" or "scale up", classify it as "zoom in"
- If the message contains words "zoom out" or "scale down", classify it as "zoom out"
- If the message contains words "rotate left" or "turn left", classify it as "rotate left"
- If the message contains words "rotate right" or "turn right", classify it as "rotate right"
- If the message contains words "picture" or "image", classify it as "show image"
- If the message contains words "current step" or "what step", classify it as "current step"
- For anything else, classify it as "repeat"

Message: "{user_message}"

Respond with ONLY the classification category."""

        try:
            result = await call_llm_with_timeout(
                self.agent.classify_regular_msg_llm, [HumanMessage(content=prompt)]
            )

            # Parse simple text response
            response_text = result.content.strip().lower()

            # Map response to message type
            if "next" in response_text:
                message_type = "next"
            elif "prev" in response_text:
                message_type = "prev"
            elif "reset" in response_text:
                message_type = "reset"
            elif "zoom in" in response_text:
                message_type = "zoom in"
            elif "zoom out" in response_text:
                message_type = "zoom out"
            elif "show image" in response_text:
                message_type = "show image"
            elif "current step" in response_text:
                message_type = "current step"
            elif "rotate left" in response_text:
                message_type = "rotate left"
            elif "rotate right" in response_text:
                message_type = "rotate right"
            else:
                message_type = "repeat"

            state.message_type = message_type
        except TimeoutError as e:
            print(f"Timeout classifying regular instruction: {e}")
            # Simple fallback based on keywords in user message
            user_lower = user_message.lower()
            if "next" in user_lower:
                state.message_type = "next"
            elif "prev" in user_lower:
                state.message_type = "prev"
            elif "reset" in user_lower:
                state.message_type = "reset"
            else:
                state.message_type = "repeat"
            raise  # Re-raise to be handled by the parent handler
        except Exception as e:
            print(f"Error classifying regular instruction: {e}")
            # Simple fallback based on keywords in user message
            user_lower = user_message.lower()
            if "next" in user_lower:
                state.message_type = "next"
            elif "prev" in user_lower:
                state.message_type = "prev"
            elif "reset" in user_lower:
                state.message_type = "reset"
            else:
                state.message_type = "repeat"

        # Don't set step completion status here - this is just message classification
        return state

    # Routing Methods
    async def route_by_message_type(self, state: AssemblyState) -> str:
        """Route based on message type classification"""
        message_type = getattr(state, "message_type", "regular")
        print(f"🔀 Routing message_type: '{message_type}'")
        if message_type == "compare":
            print("🔀 → Going to compare_assembly")
            return "compare_assembly"
        else:
            print("🔀 → Going to classify_regular_instruction")
            return "classify_regular_instruction"

    async def route_by_regular_instruction(self, state: AssemblyState) -> str:
        """Route based on regular instruction classification"""
        message_type = getattr(state, "message_type", "next")
        route_map = {
            "next": "next_step",
            "prev": "prev_step",
            "reset": "reset_assembly",
            "zoom in": "zoom_in",
            "zoom out": "zoom_out",
            "rotate left": "rotate_left",
            "rotate right": "rotate_right",
            "show image": "show_image",
            "current step": "current_step",
        }

        return route_map.get(message_type, "repeat_instruction")

    # Step Navigation Nodes
    async def next_step(self, state: AssemblyState) -> AssemblyState:
        """Move to next step"""
        if state.current_step < state.max_steps:
            # state.current_step += 1  # Unity handles the increment
            state.feedback = f"Moving to next step"
            state.is_step_complete = True  # Tell Unity to execute step change
        else:
            state.feedback = "Already at the last step"
            state.is_step_complete = False
        return state

    async def prev_step(self, state: AssemblyState) -> AssemblyState:
        """Move to previous step"""
        if state.current_step > 1:
            # state.current_step -= 1  # Unity handles the decrement
            state.feedback = f"Moving to previous step"
            state.is_step_complete = True  # Tell Unity to execute step change
        else:
            state.feedback = "Already at the first step"
            state.is_step_complete = False
        return state

    async def reset_assembly(self, state: AssemblyState) -> AssemblyState:
        """Reset to step 1"""
        state.current_step = 1
        state.assembly_history = []
        state.feedback = "Assembly reset to step 1"
        return state

    # Utility Nodes
    async def zoom_in(self, state: AssemblyState) -> AssemblyState:
        """Handle zoom in request"""
        has_multiple_refs = (
            hasattr(state, "has_multiple_images") and state.has_multiple_images
        )
        if has_multiple_refs:
            state.feedback = f"For step {state.current_step}, I have multiple reference images including close-up views. Please take a closer, detailed image of your current assembly focusing on the connection points and small pieces."
        else:
            state.feedback = "Please provide a closer image of your assembly focusing on the connection points and details."
        state.is_step_complete = True  # Tell Unity to execute zoom in
        return state

    async def zoom_out(self, state: AssemblyState) -> AssemblyState:
        """Handle zoom out request"""
        has_multiple_refs = (
            hasattr(state, "has_multiple_images") and state.has_multiple_images
        )
        if has_multiple_refs:
            state.feedback = f"For step {state.current_step}, I have multiple reference images including wider overview shots. Please take a wider view showing your entire current assembly progress."
        else:
            state.feedback = "Please provide a wider view of your assembly showing the overall structure."
        state.is_step_complete = True  # Tell Unity to execute zoom out
        return state

    async def rotate_left(self, state: AssemblyState) -> AssemblyState:
        """Handle rotate left request"""
        has_multiple_refs = (
            hasattr(state, "has_multiple_images") and state.has_multiple_images
        )
        if has_multiple_refs:
            state.feedback = f"For step {state.current_step}, I have reference images from multiple angles. Please take an image of your assembly from the left side - this will help me compare against the side-view reference images."
        else:
            state.feedback = (
                "Please provide an image of your assembly from the left side."
            )
        state.is_step_complete = True  # Tell Unity to execute rotate left
        return state

    async def rotate_right(self, state: AssemblyState) -> AssemblyState:
        """Handle rotate right request"""
        has_multiple_refs = (
            hasattr(state, "has_multiple_images") and state.has_multiple_images
        )
        if has_multiple_refs:
            state.feedback = f"For step {state.current_step}, I have reference images from multiple angles. Please take an image of your assembly from the right side - this will help me compare against the side-view reference images."
        else:
            state.feedback = (
                "Please provide an image of your assembly from the right side."
            )
        state.is_step_complete = True  # Tell Unity to execute rotate right
        return state

    async def show_image(self, state: AssemblyState) -> AssemblyState:
        """Show reference images"""
        # Count available reference images
        num_images = 0
        if hasattr(state, "reference_images") and state.reference_images:
            num_images += len(state.reference_images)
        if state.reference_image:
            num_images += 1

        if num_images > 0:
            if num_images == 1:
                state.feedback = (
                    f"Here's the reference image for step {state.current_step}"
                )
            else:
                state.feedback = f"Here are the {num_images} reference images for step {state.current_step}. These show different angles and details to help you complete this step accurately."
        else:
            state.feedback = (
                f"No reference images available for step {state.current_step}"
            )
        return state

    async def repeat_instruction(self, state: AssemblyState) -> AssemblyState:
        """Repeat current instruction"""
        state.feedback = f"Step {state.current_step}: {state.instruction_text or 'No instruction available'}"
        state.is_step_complete = True  # Tell Unity to execute repeat highlighting
        return state

    async def get_current_step(self, state: AssemblyState) -> AssemblyState:
        """Provide current step information using LLM"""
        try:
            results = await asyncio.to_thread(
                self.agent.instruction_collection.query,
                query_texts=[f"step {state.current_step}"],
                n_results=1,
                include=["documents", "metadatas"],
                where={"step": state.current_step},
            )

            instruction_text = "No specific instructions available"
            if results["documents"] and results["documents"][0]:
                instruction_text = results["documents"][0][0]

            prompt = f"""
            The user is asking about their current step in a LEGO assembly process.

            Current step: {state.current_step}
            Step instructions: {instruction_text}

            Provide a helpful, encouraging response that:
            1. Confirms the current step number
            2. Explains what needs to be done in this step
            3. Offers any helpful tips or guidance
            4. Is friendly and supportive

            IMPORTANT: Do NOT use emojis in your response. Keep it professional and text-only.
            Keep it concise but informative.
            """

            response = await call_llm_with_timeout(
                self.agent.llm, [HumanMessage(content=prompt)]
            )
            state.feedback = response.content

        except TimeoutError as e:
            print(f"Timeout getting current step info: {e}")
            state.feedback = f"You're currently on step {state.current_step}. The AI provider took too long to respond. Please try again or check your connection."
            raise  # Re-raise to be handled by the parent handler
        except Exception as e:
            print(f"Error getting current step info: {e}")
            state.feedback = f"You're currently on step {state.current_step}. Unable to retrieve detailed instructions at the moment."

        return state

    # Core Processing Nodes
    async def process_image(self, state: AssemblyState) -> AssemblyState:
        """Load images from directory for workflow node"""
        print("process_image started")
        await asyncio.sleep(0.5)

        try:
            from utils.find_seq_files import find_sequential_files
            from utils.load_image import load_image

            sequential_groups = await find_sequential_files(self.agent)
            if sequential_groups and len(sequential_groups) > 0:
                print(f"Found {len(sequential_groups)} groups:")
                for i, group in enumerate(sequential_groups):
                    print(f"  Group {i + 1}: {[f.name for f in group]}")

                # Use the last group (most recent images)
                latest_group = sequential_groups[-1]
                print(f"Processing latest group with {len(latest_group)} images")

                image_tasks = [load_image(self.agent, path) for path in latest_group]
                images_base64 = [
                    img for img in await asyncio.gather(*image_tasks) if img is not None
                ]

                if images_base64:
                    state.current_images = images_base64
                    print(f"Loaded {len(images_base64)} images successfully")
                else:
                    state.current_images = []
                    print("No valid images found")
            else:
                state.current_images = []
                print("No sequential groups found")

        except Exception as e:
            print(f"Error loading images: {e}")
            state.current_images = []

        return state

    async def retrieve_instructions(self, state: AssemblyState) -> AssemblyState:
        """Retrieve relevant instructions and reference images using RAG"""
        print("retrieve_instructions started")

        try:
            print(f"Retrieving instructions for step: {state.current_step}")

            # Use the database service's async query method
            results = await self.agent.database_service.query_instructions(
                state.current_step, 1
            )

            if results["documents"] and results["documents"][0]:
                state.instruction_text = results["documents"][0][0]

                if results["metadatas"] and results["metadatas"][0]:
                    metadata = results["metadatas"][0][0]
                    print(
                        f"Retrieved metadata for step: {metadata.get('step', 'unknown')}"
                    )

                    # Get primary reference image
                    if "reference_image" in metadata:
                        state.reference_image = metadata["reference_image"]
                        print(
                            f"Primary reference image found for step {metadata.get('step', 'unknown')}"
                        )
                    else:
                        print("No primary reference_image in metadata")

                    # Get multiple reference images
                    if "reference_images" in metadata and metadata["reference_images"]:
                        state.reference_images = metadata["reference_images"]
                        print(
                            f"Found {len(state.reference_images)} additional reference images for step {metadata.get('step', 'unknown')}"
                        )
                    else:
                        state.reference_images = []
                        print("No additional reference images found")

                    # Check if this step has multiple reference images
                    state.has_multiple_images = metadata.get(
                        "has_multiple_images", False
                    )
                    print(f"Has multiple images: {state.has_multiple_images}")

            else:
                state.instruction_text = (
                    f"Complete step {state.current_step} of the LEGO assembly."
                )
                state.reference_images = []
                state.has_multiple_images = False

        except Exception as e:
            print(f"Error retrieving instructions: {e}")
            state.instruction_text = (
                f"Complete step {state.current_step} of the LEGO assembly."
            )
            state.reference_images = []
            state.has_multiple_images = False

        return state

    async def compare_assembly(self, state: AssemblyState) -> AssemblyState:
        """Compare current assembly with reference images and instructions"""
        if (
            not state.current_images
            or len(state.current_images) == 0
            or not state.instruction_text
        ):
            return state

        # Determine how many reference images we have
        reference_images = []
        if hasattr(state, "reference_images") and state.reference_images:
            reference_images = state.reference_images
        elif state.reference_image:
            reference_images = [state.reference_image]

        # Require multiple images for proper comparison
        if len(state.current_images) == 1:
            num_ref_images = len(reference_images)
            if num_ref_images > 1:
                state.feedback = f"I have {num_ref_images} reference images showing different angles for step {state.current_step}, but I only see 1 image from you. Please take 2-3 more pictures from different angles (front, side, top view) so I can give you accurate feedback by comparing against all the reference views."
            else:
                state.feedback = "Please take more pictures from different angles. I need at least 2 images to properly assess your assembly progress."
            state.is_step_complete = False
            return state

        num_ref_images = len(reference_images)
        total_images = len(state.current_images) + num_ref_images

        comparison_prompt = f"""
        You are a LEGO assembly verification expert. Your task is to determine if the user's physical LEGO build matches the target design for step {state.current_step}.

        CURRENT STEP CONTEXT
        Step {state.current_step} Instructions: {state.instruction_text}

         IMAGE LAYOUT
        Images 1-{len(state.current_images)}: User's actual LEGO assembly (physical build)
        Images {len(state.current_images) + 1}-{total_images}: Reference images showing correct assembly
        Multiple viewing angles: front, back, left, right, top views available
        Use ALL reference angles to understand the complete 3D structure

        VERIFICATION CRITERIA

        PASS CONDITIONS (set "step_complete": true)
        Structural Match: User's assembly shows the same LEGO structure as references
        Piece Placement: All required pieces are in correct positions but assembly images could be taken from different angles and positions so exact placement in photo doesn't need to match reference exactly. Pay attention to overall structure and connections and be flexible with placement.
        Color Accuracy: Visible pieces match expected colors but if lighting causes minor color shifts, use best judgment. Transparent pieces are acceptable if they match the intended piece type.
        Connection Integrity: Pieces appear properly connected and stable
        Angle Flexibility: User's photo angle doesn't need to match reference exactly
        Confidence Threshold: 75%+ confidence that assembly is structurally correct

        FAIL CONDITIONS (set "step_complete": false)
        
        No LEGO Content: User images don't show actual physical LEGO pieces
        Missing Components: Required pieces are absent or incorrectly placed
        Structural Differences: Assembly doesn't match the target design
        Unclear Documentation: Photos are too blurry/dark to verify assembly

        ANALYSIS APPROACH

        Reference Understanding: Study all {num_ref_images} reference images to build complete mental model of target assembly
        User Assembly Review: Examine user's photos for structural elements, piece placement, and connections
        Spatial Reasoning: Account for different camera angles - focus on underlying structure, not photo perspective
        Component Verification: Check that all visible pieces match expected colors, sizes, and placements
        Previous Steps Context: Consider a previous assembly step which LEGO pieces might already be in place and help understand what pieces are relevant for the current step.
        
        RESPONSE REQUIREMENTS
        
        Tone: Encouraging and constructive - acknowledge effort and provide specific guidance
        Length: Maximum 100 words
        Specificity: If incomplete, clearly identify what pieces/connections are missing or incorrect
        Action Items: Provide concrete next steps when assembly needs correction
        Format: Do NOT use emojis. Keep response professional and text-only.

        DECISION EXAMPLES

        Different angle, correct build** → "step_complete": true
        Missing 2x4 blue brick on left side** →  "step_complete": false
        Pieces present but not connected** → "step_complete": false
        Screenshot instead of physical LEGO** →  "step_complete": false

        JSON Response:
        {{
            "step_complete": true/false,
            "missing_pieces": ["specific pieces missing"],
            "incorrect_placements": ["specific placement issues"],
            "required_actions": ["specific actions needed"],
            "confidence": 1-10
        }}
        """

        try:
            message_content = [{"type": "text", "text": comparison_prompt}]
            for image_base64 in state.current_images:
                message_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    }
                )

            # Add all reference images to the message
            if reference_images:
                print(f"Adding {len(reference_images)} reference images to comparison")
                for ref_image_base64 in reference_images:
                    message_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": ref_image_base64,
                            },
                        }
                    )
            else:
                print("No reference images available for comparison")

            messages = [HumanMessage(content=message_content)]
            response = await call_llm_with_timeout(self.agent.llm, messages)

            print(f"Raw AI response: {response.content}")

            try:
                import json

                content = response.content
                json_start = content.find("{")
                json_end = content.rfind("}") + 1

                if json_start != -1 and json_end != -1:
                    json_content = content[json_start:json_end]
                    comparison_result = json.loads(json_content)
                else:
                    raise json.JSONDecodeError("No JSON found", content, 0)

                state.comparison_result = comparison_result

                # Validate: step cannot be complete if there are issues identified
                # has_issues = (
                #     comparison_result.get("missing_pieces", []) or
                #     comparison_result.get("incorrect_placements", [])
                # )
                ai_says_complete = comparison_result.get("step_complete", False)
                state.is_step_complete = ai_says_complete

                if ai_says_complete:
                    print(
                        f"⚠️ AI inconsistency detected: marked complete but has issues. Overriding to incomplete."
                    )

                print(f"Step complete status: {state.is_step_complete}")

            except json.JSONDecodeError:
                print("JSON parsing failed, using fallback")
                state.comparison_result = {
                    "step_complete": False,
                    "missing_pieces": [],
                    "incorrect_placements": [],
                    "required_actions": ["Please review the current assembly"],
                    "confidence": 5,
                }
                state.is_step_complete = False

        except TimeoutError as e:
            print(f"Timeout comparing assembly: {e}")
            state.comparison_result = {
                "step_complete": False,
                "missing_pieces": [],
                "incorrect_placements": [],
                "required_actions": [
                    "Unable to analyze assembly due to timeout. Please try again."
                ],
                "confidence": 0,
            }
            state.is_step_complete = False
            raise  # Re-raise to be handled by the parent handler
        except Exception as e:
            print(f"Error comparing assembly: {e}")
            state.comparison_result = {
                "step_complete": False,
                "missing_pieces": [],
                "incorrect_placements": [],
                "required_actions": ["Error analyzing assembly. Please try again."],
                "confidence": 0,
            }
            state.is_step_complete = False

        return state

    def should_continue_or_provide_feedback(self, state: AssemblyState) -> str:
        """Determine the next step based on comparison results"""
        if not state.comparison_result:
            return "continue"

        if state.is_step_complete:
            return "step_complete"
        elif state.comparison_result.get("confidence", 0) < 7:
            return "feedback_needed"
        else:
            return "feedback_needed"

    async def generate_feedback(self, state: AssemblyState) -> AssemblyState:
        """Generate helpful feedback based on comparison results"""
        print("generate_feedback started")

        if not state.comparison_result:
            state.feedback = (
                "Please ensure you're following the instructions carefully."
            )
            return state

        # Check if this step has multiple reference images
        has_multiple_refs = (
            hasattr(state, "has_multiple_images") and state.has_multiple_images
        )

        feedback_prompt = f"""
        Generate helpful, encouraging feedback for a LEGO builder based on this assessment:

        Step: {state.current_step}
        Instructions: {state.instruction_text}
        Missing pieces: {state.comparison_result.get("missing_pieces", [])}
        Incorrect placements: {state.comparison_result.get("incorrect_placements", [])}
        Required actions: {state.comparison_result.get("required_actions", [])}
        {"Multiple reference images available: This step has detailed reference images from different angles" if has_multiple_refs else "Single reference image available"}

        Provide clear, step-by-step guidance that is:
        1. Encouraging and positive
        2. Specific about what to do next
        3. Easy to understand
        4. Focused on the most important issue first
        {"5. If more images are needed, suggest specific angles (front, back, side, top, close-up of connections)" if has_multiple_refs else ""}

        {"IMPORTANT: If the user needs to take better photos, mention that multiple reference angles are available for this step and suggest taking photos from similar angles for better comparison." if has_multiple_refs else ""}

        IMPORTANT: Do NOT use emojis in your response. Keep it professional and text-only.
        Keep it concise but helpful.
        """

        try:
            response = await call_llm_with_timeout(
                self.agent.llm, [HumanMessage(content=feedback_prompt)]
            )
            state.feedback = response.content
        except TimeoutError as e:
            print(f"Timeout generating feedback: {e}")
            state.feedback = "The AI provider took too long to respond. Please check your assembly against the instructions and try again."
            raise  # Re-raise to be handled by the parent handler
        except Exception as e:
            print(f"Error generating feedback: {e}")
            state.feedback = "Please review the instructions and try again."

        return state

    async def complete_step(self, state: AssemblyState) -> AssemblyState:
        """Handle step completion and increment to next step"""
        # Record completion of current step
        state.assembly_history.append(
            {
                "step": state.current_step,
                "completed": True,
                "pieces_used": state.detected_pieces.copy(),
            }
        )

        current_completed_step = state.current_step
        if state.current_step < state.max_steps:
            # state.current_step += 1
            state.feedback = f"Great job! Step {current_completed_step} completed successfully. You're now on step {state.current_step + 1} !"
        else:
            state.feedback = (
                "Congratulations! You've completed the entire LEGO assembly!"
            )

        return state

    async def update_progress(self, state: AssemblyState) -> AssemblyState:
        """Update progress without completing the step"""
        state.assembly_history.append(
            {
                "step": state.current_step,
                "completed": False,
                "pieces_used": state.detected_pieces.copy(),
                "feedback_given": state.feedback,
            }
        )
        return state
