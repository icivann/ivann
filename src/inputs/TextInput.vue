<template>
  <span class="TextInput">
    <span class="form-group">
      <label
        v-if="label !== ''"
        class="input-label"
      >
        {{ label }}
      </label>

      <input
        :type="type"
        class="form-control"
        :class="{ 'is-invalid': validation && validation.$error }"
        :value="value"
        @input="$emit('input', $event.target.value)"
        :placeholder="placeholder"
        :disabled="disabled"
      />

      <ValidationErrors
        v-if="validation && validation.$error"
        :validation="validation"
      />
    </span>
  </span>
</template>

<script lang="ts">
import { Validation } from 'vuelidate';
import { Component, Prop, Vue } from 'vue-property-decorator';
import ValidationErrors from '@/inputs/ValidationErrors.vue';

@Component({
  components: {
    ValidationErrors,
  },
})
export default class TextInput extends Vue {
  @Prop({ default: '' }) readonly label!: string;
  @Prop({ default: false }) readonly password!: boolean;
  @Prop({ required: true }) readonly placeholder!: string;
  @Prop({ required: true }) readonly validation!: Validation;
  @Prop({ default: false }) readonly disabled!: boolean;
  @Prop({ default: '' }) readonly value!: string;

  get type() {
    if (this.password) {
      return 'password';
    }
    return 'text';
  }
}
</script>
