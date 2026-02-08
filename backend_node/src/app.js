import express from "express"
import dotenv from "dotenv"
import cors from "cors"
dotenv.config()
const app=express()
app.use(express.json())
import connection from "./db/connectDB.js"
import userRouter from "./routes/user.routes.js"
import topicRouter from "./routes/topic.routes.js"
app.use("/users",userRouter)
app.use("/topic",topicRouter)
app.use(
  cors({
    origin: [
      "http://localhost:8000",
    ],
    credentials: true,
  })
);
connection()
.then(()=>{
    app.listen(3000,()=>{
        console.log("Server Running on port 3000");
    });
})
.catch((error)=>{
    throw error
})