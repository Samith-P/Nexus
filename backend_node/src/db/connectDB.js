import mongoose from "mongoose"

const connection=async ()=>{
    try{
        const connectdb=await mongoose.connect(process.env.MONGODB_URL+"/Nexus")
        return connectdb
    }
    catch(error){
        throw error
    }
}

export default connection