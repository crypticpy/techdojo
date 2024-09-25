def display_troubleshooting_page_sync(ticket_data: Dict[str, Any], prediction_result: Dict[str, Any]):
    """
    Display a simplified troubleshooting page for debugging purposes.
    """
    try:
        st.write("## Troubleshooting Chat (Debug Mode)")

        st.write("### Ticket Data")
        st.json(ticket_data)

        st.write("### Prediction Result")
        st.json(prediction_result)

        st.write("### Debug Information")
        st.write(f"Session State Keys: {list(st.session_state.keys())}")

        # Ensure the agent is initialized
        if 'agent' not in st.session_state or st.session_state.agent is None:
            st.write("Initializing agent...")
            model = st.session_state.get('selected_model', 'gpt-4o-2024-08-06')
            temperature = st.session_state.get('options', {}).get('Temperature', 0.1)
            incident_context = f"""You are an AI assistant helping with a technical support issue. 
            Incident details: {ticket_data}
            Always use the Perplexity Search tool when asked to look up information or when you need to provide specific technical details.
            Maintain context throughout the conversation and avoid repeating previous responses."""
            try:
                st.session_state.agent = setup_agent(model, temperature, incident_context)
                st.write("Agent initialized successfully")
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")

        if 'agent' in st.session_state:
            st.write("Agent is initialized")
            st.write(f"Agent type: {type(st.session_state.agent)}")
        else:
            st.write("Agent is not initialized")

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        st.write(f"Number of messages: {len(st.session_state.messages)}")

        st.write("### Test Input")
        user_input = st.text_input("Enter a test message")
        if st.button("Send"):
            st.write(f"You entered: {user_input}")
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.write(f"Updated number of messages: {len(st.session_state.messages)}")

            # Simulate agent response
            if 'agent' in st.session_state and st.session_state.agent is not None:
                try:
                    agent_response = st.session_state.agent.run(user_input)
                    st.write(f"Agent response: {agent_response}")
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    st.error(f"Error getting agent response: {str(e)}")
            else:
                st.error("Agent is not initialized, cannot get response")

        st.write("### Chat History")
        for msg in st.session_state.messages:
            st.write(f"{msg['role']}: {msg['content']}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in display_troubleshooting_page_sync: {str(e)}", exc_info=True)