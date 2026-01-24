import { User } from "../db/user.model.js";

const generateRefreshAccessToken = async (user) => {
  const reft = await user.generateRefreshToken();
  const act = await user.generateAccessToken();

  user.refreshToken = reft;
  await user.save({ validateBeforeSave: false });

  return { act, reft };
};

const registerUser = async (req, res) => {
  const { email, password, fullName,collegeName,collegeID,Degree,GraduationYear} = req.body;
  

  if (!email || !password || !fullName || !collegeName || !collegeID || !Degree || !GraduationYear) {
    return res.status(400).json({
      message: "All fields are required",
    });
  }

  const existingUser = await User.findOne({ email });
  if (existingUser) {
    return res.status(409).json({
      message: "User already exists",
    });
  }

  const user = await User.create({ email, password, fullName,collegeName,collegeID,Degree,GraduationYear });
  const createduser=await User.findById(user._id).select("-password -refreshToken")
  return res.status(201).json({
    user: createduser,
  });
};

const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    console.log(process.env.MONGODB_URL)
    if (!email || !password) {
      return res.status(400).json({ message: "Enter both Email and Password" });
    }

    const user = await User
    .findOne({ email })
    .select("+password");

    if (!user) {
      return res.status(401).json({ message: "User does not exist" });
    }

    const isvalidpass = await user.isPasswordCorrect(password);
    if (!isvalidpass) {
      return res.status(401).json({ message: "Password incorrect" });
    }

    const { act, reft } = await generateRefreshAccessToken(user);

    const options = {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
    };

    const createduser = await User
      .findById(user._id)
      .select("-password -refreshToken");

    return res
      .status(200)
      .cookie("accesstoken", act, options)
      .cookie("refreshToken", reft, options)
      .json({ user: createduser });

  } catch (error) {
    console.error("LOGIN ERROR:", error);
    return res.status(500).json({
      message: error.message,
    });
  }
};

const logoutUser = async (req, res) => {
  try {
    const refreshToken = req.cookies?.refreshToken;
    if (refreshToken) {
      await User.findOneAndUpdate(
        { refreshToken },
        { $unset: { refreshToken: 1 } }
      );
    }
    const options = {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
    };
    return res
      .clearCookie("accesstoken", options)
      .clearCookie("refreshToken", options)
      .status(200)
      .json({ message: "Logged out" });

  } catch (err) {
    return res.status(500).json({ message: "Logout failed" });
  }
};

export { registerUser, loginUser ,logoutUser};
