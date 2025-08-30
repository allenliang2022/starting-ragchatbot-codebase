from openai import OpenAI
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content, lessons, or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson lists, course links, or when users ask for an "outline" or "overview" of a course
- **Sequential tool usage allowed**: Up to 2 rounds of tool calls to gather comprehensive information
- **Round 1**: Use initial tools to gather primary information
- **Round 2**: If needed, use additional tools to clarify, expand, or gather complementary information
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Sequential Decision Making:
- After each round, assess if you have sufficient information to provide a complete answer
- If information is incomplete or you need clarification, proceed to the next round
- Use the second round strategically for follow-up searches or outline requests
- Provide final comprehensive response after gathering sufficient information

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tools first, then answer
- **Complex queries**: May require multiple tool calls across rounds for comprehensive coverage
- **Course outline questions**: Use get_course_outline tool to provide course title, course link, and complete lesson list
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def _convert_tools_to_openai_format(self, anthropic_tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool format to OpenAI tool format"""
        openai_tools = []
        
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling up to max_rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize messages for the conversation
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Sequential round loop
        for round_num in range(max_rounds):
            # Make API call for this round
            response = self._make_api_call(messages, tools)
            
            # Check if AI wants to use tools
            if response.choices[0].finish_reason == "tool_calls" and tool_manager:
                # Handle tool execution and check if we should continue
                should_continue, final_response = self._handle_tool_execution_round(
                    response, messages, tool_manager, round_num + 1, max_rounds
                )
                
                if not should_continue:
                    # Return final response if AI decided to stop or we have a direct response
                    return final_response if final_response else self._get_final_response(messages)
                    
                # Continue to next round if we haven't hit the limit
                if round_num + 1 >= max_rounds:
                    # Force final response on last round
                    return self._get_final_response(messages)
            else:
                # No tools called, return direct response
                return response.choices[0].message.content
        
        # Fallback - should not reach here, but return final response just in case
        return self._get_final_response(messages)

    def _make_api_call(self, messages: List[Dict], tools: Optional[List] = None):
        """Make API call with current messages and tools"""
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"
        
        return self.client.chat.completions.create(**api_params)

    def _get_final_response(self, messages: List[Dict]) -> str:
        """Get final response without tools"""
        # Add continuation prompt to encourage final synthesis
        continuation_messages = messages + [{
            "role": "user", 
            "content": "Based on all the information gathered, please provide your final comprehensive answer."
        }]
        
        final_params = {
            **self.base_params,
            "messages": continuation_messages
        }
        
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content

    def _handle_tool_execution_round(self, response, messages: List[Dict], tool_manager, current_round: int, max_rounds: int) -> tuple[bool, Optional[str]]:
        """
        Handle tool execution for a round and determine if we should continue.
        
        Args:
            response: The API response containing tool calls
            messages: Current conversation messages
            tool_manager: Manager to execute tools
            current_round: Current round number (1-based)
            max_rounds: Maximum allowed rounds
            
        Returns:
            Tuple of (should_continue: bool, final_response: Optional[str])
            If should_continue is False, final_response contains the AI's response
        """
        # Add AI's tool use request to messages
        messages.append(response.choices[0].message)
        
        # Execute all tool calls and collect results
        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            try:
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name,
                    **json.loads(tool_call.function.arguments)
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            except Exception as e:
                # Handle tool execution errors gracefully
                error_msg = f"Tool execution failed: {str(e)}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg
                })
        
        # If this is the last round, don't continue
        if current_round >= max_rounds:
            return False, None
        
        # Ask AI if it wants to continue with more tools
        return self._should_continue_to_next_round(messages)

    def _should_continue_to_next_round(self, messages: List[Dict]) -> tuple[bool, Optional[str]]:
        """
        Ask AI if it wants to continue to the next round or provide final answer.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            Tuple of (should_continue: bool, final_response: Optional[str])
            If should_continue is False, final_response contains the AI's decision
        """
        # Add continuation decision prompt
        decision_messages = messages + [{
            "role": "user",
            "content": "Based on the information you've gathered, do you have sufficient information to provide a complete answer, or do you need to use additional tools? If you need more information, use the appropriate tools. If you have sufficient information, provide your final comprehensive response."
        }]
        
        decision_params = {
            **self.base_params,
            "messages": decision_messages
        }
        
        # Make API call to get AI's decision
        decision_response = self.client.chat.completions.create(**decision_params)
        
        # If AI wants to use more tools, continue; otherwise finalize
        if decision_response.choices[0].finish_reason == "tool_calls":
            # AI wants to use more tools - add this decision to messages for next round
            messages.append(decision_response.choices[0].message)
            return True, None
        else:
            # AI provided final response - return it directly
            return False, decision_response.choices[0].message.content
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        Legacy method for backward compatibility - single round only.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append(initial_response.choices[0].message)
        
        # Execute all tool calls and collect results
        tool_calls = initial_response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content