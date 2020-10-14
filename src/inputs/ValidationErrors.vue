<template>
  <div class="ValidationErrors">
    <!-- Required -->
    <ValidationError
      v-if="isInvalid(validation['required'])"
      text="Field is required."
    />

    <!-- Min Length -->
    <ValidationError
      v-if="isInvalid(validation['minLength'])"
      :text="'Must have at least ' + validation.$params.minLength.min + ' characters.'"
    />

    <!-- Max Length -->
    <ValidationError
      v-if="isInvalid(validation['minLength'])"
      :text="'Must have less than ' + validation.$params.maxLength.max + ' characters.'"
    />

    <!-- Email -->
    <ValidationError
      v-if="isInvalid(validation['email'])"
      text="Must be a valid email."
    />
  </div>
</template>

<script lang="ts">
import { Validation } from 'vuelidate';
import { Component, Prop, Vue } from 'vue-property-decorator';
import ValidationError from '@/inputs/ValidationError.vue';

@Component({
  components: {
    ValidationError,
  },
})
export default class ValidationErrors extends Vue {
  @Prop({ required: true }) readonly validation!: Validation;

  private defined = (obj: unknown) => typeof obj !== 'undefined';

  private isInvalid(validator: unknown) {
    return this.defined(validator) && !validator;
  }
}
</script>
