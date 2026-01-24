import express from "express"
import dotenv from "dotenv"
dotenv.config()
const app=express()
app.use(express.json())
import connection from "./db/connectDB.js"
import userRouter from "./routes/user.routes.js"
app.use("/users",userRouter)

connection()
.then(()=>{
    app.listen(3000,()=>{
        console.log("Server Running on port 3000");
    });
})
.catch((error)=>{
    throw error
})