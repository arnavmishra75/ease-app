var socket = io();

// Listen for reset event and refresh/reset variables
socket.on('reset_hume', function() {
    document.getElementById('chatbox').innerHTML = ''; // Clear chat UI
});

// Function to manually reset conversation
function resetConversation() {
    window.location.href = "/new_conversation";  // Triggers Flask reset
}