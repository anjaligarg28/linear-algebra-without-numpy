import streamlit as st
import numpy as np
import copy
from linear_algebra import LinearAlgebra
import pandas as pd

st.set_page_config(layout="wide") # Set the page configuration to wide mode
st.markdown( #  Add custom CSS for padding
    """
    <style>
    .main {
        background-color: #222221; /* Set the background color to gray */
        color: white; /* Set the text color to white for better contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def perform_operation(operation, matrix):

    matrix = copy.deepcopy(matrix)
    linearAlgebra = LinearAlgebra()

    if(operation == "Transpose"):
        st.write(np.array(linearAlgebra.transpose(matrix)))

    elif(operation == "Determinant"):
        st.write(f"Determinant of the given matrix is: {linearAlgebra.determinant(matrix)}")

    elif(operation == "Trace"):
        st.write(f"Trace of the given matrix is: {linearAlgebra.trace(matrix)}")

    elif(operation == "Inverse"):
        st.write(np.array(linearAlgebra.inverse_using_cofactors(matrix)))

    elif(operation == "Row Echelon Form"):
        st.write(np.array(linearAlgebra.row_echelon_form(matrix)))

    elif(operation == "Rank"):
        st.write(f"Rank of the given matrix is: {linearAlgebra.rank(matrix)}")

    elif(operation == "Norms"):
        orders = [1,2,-1,"inf","-inf"]
        for ord in orders:
            st.write(f"{ord} Norm: {linearAlgebra.norm(matrix, ord)}")

    elif(operation == "LU Decomposition"):
        L, U = linearAlgebra.lu_decomposition(matrix)
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("Lower Triangular Matrix: ")
            st.write(np.array(L))
        with col2:
            st.write("Upper Triangular Matrix: ")
            st.write(np.array(U))

    elif(operation == "QR Decomposition"):
        Q, R = linearAlgebra.qr_decomposition(matrix)
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("Orthogonal Matrix: ")
            st.write(np.array(Q))
        with col2:
            st.write("Right Triangular Matrix: ")
            st.write(np.array(R))

    elif(operation == "Eigen Values"):
        st.write(f"Eigen values of the given matrix are: {np.array(linearAlgebra.eigen_values(matrix))}")

def main():
    # Center-aligned title using HTML and CSS
    st.markdown("<h1 style='text-align: center; color: deepskyblue;'>Linear Algebra Operations without NumPy</h1>", unsafe_allow_html=True)
    s1, inp, space, out, s2 = st.columns([0.1,1,0.1,1, 0.1])

    with inp:
        st.write("## Enter dimension and operation")

        rowinp, colinp = st.columns([1,1])
        # Input for the number of rows and columns
        with rowinp:
            rows = st.number_input('Enter the number of rows:', min_value=1, value=3)
        with colinp:
            cols = st.number_input('Enter the number of columns:', min_value=1, value=3)

        operations_available = [
            'Transpose',
            'Determinant',
            'Trace',
            'Inverse',
            'Row Echelon Form',
            'Rank',
            'Norms',
            'LU Decomposition',
            'QR Decomposition',
            'Eigen Values',
        ]
        # Select the matrix operation
        operation = st.selectbox('Select an operation:', operations_available)

    with out:
        st.write("## Enter matrix values:")
        # Initialize or reset the matrix in session state
        if 'matrix' not in st.session_state or st.session_state.matrix.shape != (rows, cols):
            st.session_state.matrix = np.zeros((rows, cols), dtype=int)

        # Creating the matrix grid for user input
        matrix_data = []
        for r in range(rows):
            row_data = []
            columns = st.columns(cols)  # Create a column for each element in the row
            for c in range(cols):
                with columns[c]:  # Use the column context
                    # Use number input for each matrix cell, ensuring each cell has a unique key and providing a visible label
                    label = f'Value at ({r+1}, {c+1}):'
                    val = st.number_input(label, value=st.session_state.matrix[r, c], key=f'{r}-{c}', label_visibility='collapsed')
                    row_data.append(val)
            matrix_data.append(row_data)

        # Update the session state matrix after collecting all inputs
        st.session_state.matrix = np.array(matrix_data) 
        button = st.button('Perform Operation')

    st.write("---")
    s4, inpDisplay, s5, outDisplay, s6 = st.columns([0.1,1,0.1,1, 0.1])
    with inpDisplay:
        st.write("Input Matrix:")
        st.write(np.array(matrix_data))
    with outDisplay:
        # Perform the selected operation and display the result
        if button:
            st.write('Result:')
            perform_operation(operation, matrix_data)

if __name__ == '__main__':
    main()
