const topicselection = async (req, res) => {
  const url = "http://localhost:8000/topic-recommendation"; // replace with real API

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req.body), 
    });
    if (!response.ok) {
      return res.status(response.status).json({
        success: false,
        message: "External API failed",
      });
    }
    const result = await response.json();

    return res.status(200).json({
      success: true,
      data: result,
    });

  } catch (error) {
    console.error("Error creating post:", error);
    return res.status(500).json({
      success: false,
      message: "Server error",
    });
  }
};

export { topicselection };
