import mongoose, {Mongoose, Schema} from "mongoose";
import bcrypt from "bcrypt"
import jwt from "jsonwebtoken"

const userSchema = mongoose.Schema({
    email:{
        type:String,
        required: true,
        unique:true,
        index:true,
        trim:true,
        lowercase:true 
    },
    fullName:{
        type:String,
        required:true, 
        trim:true
    },
    password:{
        type:String,
        required:true,
        trim:true
    },
    papersCreated:{
        type:Schema.Types.ObjectId,
        ref:"papers"
    },
    collegeName:{
        type:String,
        required:true,
        trim:true
    },
    collegeID:{
        type:String,
        required:true,
        trim:true
    },
    Degree:{
        type:String,
        required:true,
        trim:true
    },
    GraduationYear:{
        type:Number,
        required:true
    }
})

userSchema.pre("save", async function () {
    if (!this.isModified("password")) return
    this.password = await bcrypt.hash(this.password, 10)
})

userSchema.methods.isPasswordCorrect = async function (password) {
    return await bcrypt.compare(password, this.password)
}

userSchema.methods.generateAccessToken = function () {
    return jwt.sign(
        {
            _id: this._id,
            email: this.email,
            fullname: this.fullname,
        },
        process.env.ACCESS_TOKEN_SECRET,
        { expiresIn: process.env.ACCESS_TOKEN_EXPIRY }
    )
}

userSchema.methods.generateRefreshToken = function () {
    return jwt.sign(
        { _id: this._id },
        process.env.REFRESH_TOKEN_SECRET,
        { expiresIn: process.env.REFRESH_TOKEN_EXPIRY }
    )
}



const User=mongoose.model("User",userSchema)
export {User}